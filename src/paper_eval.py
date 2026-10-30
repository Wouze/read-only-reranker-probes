#!/usr/bin/env python3
"""Frozen R1 cascade: exact-reuse forward, wall-clock, and one-shot public scores.

The rule file is hashed before dev or DL19 is read. Check mode never opens those files.
Survivors of the last stage are continued from the cached document K/V, so their
margin is the frozen model's margin.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from core import Manual, Plan, flex_doc_attend, flex_doc_block, score_query  # noqa: E402
from fit_rules import ndcg  # noqa: E402
from probe import Probe, load_engine  # noqa: E402
from spec import fit_ridge, rms  # noqa: E402
from train_qwen_draft import OUT, continuation, load_manifest, query_prefix  # noqa: E402

ROOT = OUT.parents[2]
DATA = OUT / "fast06"
RULE = DATA / "paper_r1.json"
RULE_HASH = DATA / "paper_r1.sha256"
OK = DATA / "paper_r1.ok"

# Written down before any 4B full-probe score, confirmed on check, now frozen.
RULE_BODY = {
    "name": "R1",
    "model": "Qwen/Qwen3-Reranker-4B",
    "probe": "q4_full",
    "stages": [
        {"kind": "probe", "layer": 9, "keep": 60},
        {"kind": "probe", "layer": 15, "keep": 40},
        {"kind": "ridge", "layer": 21, "keep": 15},
    ],
    "final": "exact margin of the last survivors",
}


def freeze_rule() -> None:
    text = json.dumps(RULE_BODY, indent=2, sort_keys=True) + "\n"
    RULE.write_text(text, encoding="utf-8")
    RULE_HASH.write_text(hashlib.sha256(text.encode("utf-8")).hexdigest() + "\n", encoding="utf-8")
    print(f"hashed {RULE_HASH.read_text().strip()}", flush=True)


def require_rule() -> dict:
    digest = RULE_HASH.read_text(encoding="utf-8").strip()
    if hashlib.sha256(RULE.read_bytes()).hexdigest() != digest:
        raise SystemExit("paper_r1.json does not match its hash")
    return json.loads(RULE.read_text(encoding="utf-8"))


# None leaves every GEMM on the padded tensor. "mlp" gathers real tokens for the
# SwiGLU only. "all" also gathers them for the input norm, q/k/v, and o_proj.
COMPACT = None
COUNT_PAD = False
PAD = {"real": 0, "tok": 0}


def _real_index(valid: torch.Tensor) -> torch.Tensor:
    return valid.reshape(-1).nonzero(as_tuple=False).flatten()


def _pointwise(x: torch.Tensor, valid: torch.Tensor, fn):
    """Apply a token-wise map to real rows only and scatter the result back."""
    idx = _real_index(valid)
    if idx.numel() == 0:
        return x.new_zeros(x.shape)
    h = x.reshape(-1, x.shape[-1]).index_select(0, idx)
    y = fn(h)
    out = x.new_zeros(x.shape)
    out.view(-1, x.shape[-1]).index_copy_(0, idx, y)
    return out


def _proj_real(eng, layer, x: torch.Tensor, valid: torch.Tensor, plan):
    """q, k, v for real tokens, scattered into a padded [B, H, W, hd] buffer."""
    B, W, d = x.shape
    idx = _real_index(valid)
    h = layer.input_layernorm(x.reshape(-1, d).index_select(0, idx))
    q, k, v = eng._proj(layer, h, plan)
    b = torch.div(idx, W, rounding_mode="floor")
    t = idx - b * W

    def place(real: torch.Tensor) -> torch.Tensor:
        out = real.new_zeros(B, real.shape[1], W, real.shape[2])
        out[b, :, t, :] = real
        return out

    return place(q), place(k), place(v)


# None leaves every layer on the padded tensor. "mlp" runs the SwiGLU on real
# tokens only. "all" does the same for the input norm, QKV, and the output projection.
COMPACT = None


def _real_index(valid: torch.Tensor) -> torch.Tensor:
    return valid.reshape(-1).nonzero(as_tuple=False).flatten()


def apply_real(x: torch.Tensor, valid: torch.Tensor, fn):
    """Apply a token-wise function to the unmasked rows of x. The function may change the last dimension."""
    idx = _real_index(valid)
    y = fn(x.reshape(-1, x.shape[-1]).index_select(0, idx))
    out = y.new_zeros(x.shape[0], x.shape[1], y.shape[-1])
    out.view(-1, y.shape[-1]).index_copy_(0, idx, y)
    return out


def proj_real(eng, layer, x: torch.Tensor, valid: torch.Tensor, plan):
    """Q, K, V of the real tokens, scattered back to (B, heads, W, hd)."""
    B, W, d = x.shape
    idx = _real_index(valid)
    h = layer.input_layernorm(x).reshape(-1, d).index_select(0, idx)
    q, k, v = eng._proj(layer, h[:, None, :], plan)
    q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
    b = torch.div(idx, W, rounding_mode="floor")
    t = idx - b * W

    def place(src: torch.Tensor) -> torch.Tensor:
        out = src.new_zeros(B, src.shape[1], W, src.shape[2])
        out[b, :, t, :] = src
        return out

    return place(q), place(k), place(v)


def chunks(lengths: list[int], budget: int):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * max(1, lengths[order[j]]) <= budget:
            j += 1
        j = max(j, i + 1)
        yield order[i:j]
        i = j


def advance(eng: Manual, cache, P: int, token_rows: list[list[int]], x, kvs, start: int, end: int):
    """Run layers start..end on already-embedded tokens. token_rows are unpadded ids.

    x is (B, W, d) after layer start-1, or None when start is 0.
    kvs holds layers 0..start-1, each (B, hkv, W, hd). Returns the same, through `end`.
    """
    dev = eng.device
    B = len(token_rows)
    W = max(1, max(len(t) for t in token_rows))
    ids = torch.zeros((B, W), dtype=torch.long, device=dev)
    lens = torch.tensor([len(t) for t in token_rows], device=dev)
    for i, t in enumerate(token_rows):
        if t:
            ids[i, : len(t)] = torch.tensor(t, device=dev)
    pos_base = torch.arange(W, device=dev)
    valid = pos_base[None] < lens[:, None]
    if start == 0:
        x = eng.base.embed_tokens(ids)
        kvs = []
    elif x.shape[1] != W or kvs[-1][0].shape[2] != W:
        raise RuntimeError(f"state width {x.shape[1]} != token width {W}")
    cos, sin = eng.base.rotary_emb(x, (P + pos_base)[None].expand(B, W))
    causal = torch.ones((W, W), dtype=torch.bool, device=dev).tril()[None] & valid[:, None, :]
    mask = torch.cat([torch.ones((B, W, P), dtype=torch.bool, device=dev), causal], 2)
    flex_block, flex_key = flex_doc_block(P, lens, W)
    plan = Plan()
    for j in range(start, end + 1):
        layer = eng.base.layers[j]
        if COMPACT == "all":
            q, k, v = proj_real(eng, layer, x, valid, plan)
        else:
            q, k, v = eng._proj(layer, layer.input_layernorm(x), plan)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        kvs.append((k, v))
        pk, pv = cache[j]
        kf = torch.cat([pk.expand(B, -1, -1, -1), k], 2)
        vf = torch.cat([pv.expand(B, -1, -1, -1), v], 2)
        a = flex_doc_attend(q, kf, vf, flex_block, flex_key, layer.self_attn.scaling, mask[:, None])
        joined = a.transpose(1, 2).reshape(B, W, -1)
        if COMPACT == "all":
            x = x + apply_real(joined, valid, layer.self_attn.o_proj)
        else:
            x = x + layer.self_attn.o_proj(joined)
        if COMPACT in ("mlp", "all"):
            x = x + apply_real(x, valid, lambda h, layer=layer: layer.mlp(layer.post_attention_layernorm(h)))
        else:
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = x * valid[..., None]
    return x, kvs, lens, valid


def slice_state(x, kvs, lens, idx: list[int]):
    """Unpad one batch back to per-document states, in `idx` order."""
    out = []
    for local, doc_i in enumerate(idx):
        n = int(lens[local])
        out.append((doc_i, x[local, :n].contiguous(), [(k[local, :, :n].contiguous(), v[local, :, :n].contiguous()) for k, v in kvs]))
    return out


def batch_state(states: list[tuple]):
    """states: (doc_i, x (n,d), kvs). Pad to one batch. kvs aligned to the deepest."""
    dev = states[0][1].device
    lens = [s[1].shape[0] for s in states]
    W = max(lens)
    d = states[0][1].shape[-1]
    B = len(states)
    x = torch.zeros((B, W, d), dtype=states[0][1].dtype, device=dev)
    depth = len(states[0][2])
    kvs = []
    hkv, hd = states[0][2][0][0].shape[0], states[0][2][0][0].shape[-1]
    for j in range(depth):
        k = torch.zeros((B, hkv, W, hd), dtype=states[0][2][j][0].dtype, device=dev)
        v = torch.zeros_like(k)
        kvs.append((k, v))
    for i, (_, xi, kvi) in enumerate(states):
        n = xi.shape[0]
        x[i, :n] = xi
        for j in range(depth):
            kvs[j][0][i, :, :n] = kvi[j][0]
            kvs[j][1][i, :, :n] = kvi[j][1]
    return x, kvs, lens


def probe_at(eng, probe: Probe, cache, P, kvs, lens, valid, layer: int) -> torch.Tensor:
    B, S = lens.shape[0], probe.emb.shape[0]
    hq, hd = eng.hq, eng.hd
    x = probe.emb[None].expand(B, S, -1)
    pos = P + lens[:, None] + torch.arange(S, device=lens.device)
    cos, sin = eng.base.rotary_emb(x, pos)
    W = valid.shape[1]
    mask = torch.cat([
        torch.ones((B, S, P), dtype=torch.bool, device=lens.device),
        valid[:, None, :].expand(B, S, W),
        torch.ones((S, S), dtype=torch.bool, device=lens.device).tril()[None].expand(B, S, S),
    ], 2)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        for j, ad in enumerate(probe.ad):
            if j > layer:
                break
            layer_m = eng.base.layers[j]
            attn = layer_m.self_attn
            h = layer_m.input_layernorm(x)
            q = attn.q_norm(ad.q(h).view(B, S, hq, hd)).transpose(1, 2)
            k = attn.k_norm(ad.k(h).view(B, S, -1, hd)).transpose(1, 2)
            v = ad.v(h).view(B, S, -1, hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            pk, pv = cache[j]
            dk, dv = kvs[j]
            dt = q.dtype
            kf = torch.cat([pk.expand(B, -1, -1, -1).to(dt), dk.to(dt), k.to(dt)], 2)
            vf = torch.cat([pv.expand(B, -1, -1, -1).to(dt), dv.to(dt), v.to(dt)], 2)
            a = F.scaled_dot_product_attention(q, kf, vf, attn_mask=mask[:, None], enable_gqa=True, scale=attn.scaling)
            x = x + ad.o(a.transpose(1, 2).reshape(B, S, -1))
            h2 = layer_m.post_attention_layernorm(x)
            x = x + ad.down(F.silu(ad.gate(h2)) * ad.up(h2))
            if j == layer:
                return probe.heads[str(j)](x[:, -1].float())
    raise RuntimeError(f"probe has no layer {layer}")


def suffix_through(eng, cache, P, doc_kvs, doc_lens, suffix_ids: list[int], start: int, end: int, sx):
    """Run the real suffix from layer start..end, reading document K/V. sx is None at start 0."""
    dev = eng.device
    B = doc_lens.shape[0]
    S = len(suffix_ids)
    if start == 0:
        ids = torch.tensor(suffix_ids, device=dev)[None].expand(B, S)
        sx = eng.base.embed_tokens(ids)
    pos = P + doc_lens[:, None] + torch.arange(S, device=dev)
    cos, sin = eng.base.rotary_emb(sx, pos)
    ones = torch.ones((S, S), dtype=torch.bool, device=dev).tril()
    W = doc_kvs[0][0].shape[2]
    doc_valid = torch.arange(W, device=dev)[None] < doc_lens[:, None]
    mask = torch.cat([
        torch.ones((B, S, P), dtype=torch.bool, device=dev),
        doc_valid[:, None, :].expand(B, S, W),
        ones[None].expand(B, S, S),
    ], 2)
    plan = Plan()
    for j in range(start, end + 1):
        layer = eng.base.layers[j]
        q, k, v = eng._proj(layer, layer.input_layernorm(sx), plan)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        pk, pv = cache[j]
        dk, dv = doc_kvs[j]
        kf = torch.cat([pk.expand(B, -1, -1, -1), dk, k], 2)
        vf = torch.cat([pv.expand(B, -1, -1, -1), dv, v], 2)
        a = F.scaled_dot_product_attention(q, kf, vf, attn_mask=mask[:, None], enable_gqa=True, scale=layer.self_attn.scaling)
        sx = sx + layer.self_attn.o_proj(a.transpose(1, 2).reshape(B, S, -1))
        sx = sx + layer.mlp(layer.post_attention_layernorm(sx))
    return sx


def keep_top(scores: torch.Tensor, alive: list[int], k: int) -> list[int]:
    """Keep everyone at or above the Kth score, matching spec5's >= kth rule."""
    if not alive:
        return []
    kth = sorted((float(scores[i]) for i in alive), reverse=True)[min(k, len(alive)) - 1]
    return [i for i in alive if float(scores[i]) >= kth - 1e-6]


def ridge_beta(eng_unused=None) -> torch.Tensor:
    dest = DATA / "probes" / "ridge_4b_l21.pt"
    if dest.exists():
        return torch.load(dest, map_location="cpu")["beta"]
    hid = torch.load(DATA / "runs/fit/collect_4b_b8000_hidden.pt", map_location="cpu")
    run = torch.load(DATA / "runs/fit/collect_4b_b8000.pt", map_location="cpu")
    fq = list(hid["hidden"])
    k = hid["layers"].index(21)
    X = torch.stack([rms(hid["hidden"][q][:, k].float()) for q in fq])
    y = torch.tensor([run["queries"][q]["margins"] for q in fq])
    cut = int(0.85 * len(fq))
    import numpy as np
    best = None
    for lam in (1e-4, 1e-3, 1e-2):
        beta = fit_ridge(X[:cut], y[:cut], lam)
        pred = X[cut:] @ beta
        c = float(np.mean([np.corrcoef(pred[i].numpy(), y[cut + i].numpy())[0, 1] for i in range(len(pred))]))
        if best is None or c > best[0]:
            best = (c, lam)
    beta = fit_ridge(X, y, best[1])
    torch.save({"beta": beta, "lam": best[1], "heldout_corr": best[0]}, dest)
    print(f"ridge layer 21 lam {best[1]} held-out corr {best[0]:.3f}", flush=True)
    return beta


class Cascade:
    def __init__(self, eng: Manual, probe: Probe, beta: torch.Tensor, suffix_ids: list[int], budget: int = 4000):
        self.eng = eng
        self.probe = probe
        self.beta = beta.to(eng.device)
        self.suffix_ids = suffix_ids
        self.budget = budget
        self.stages = [(9, 60), (15, 40)]
        self.ridge_layer = 21
        self.ridge_keep = 15

    def _last(self, x, lens) -> torch.Tensor:
        rows = torch.arange(x.shape[0], device=x.device)
        return x[rows, lens - 1]

    def margins_of(self, x, lens) -> torch.Tensor:
        h = self.eng.base.norm(self._last(x, lens)).float()
        return h @ self.eng.w

    def ridge_of(self, x, lens) -> torch.Tensor:
        return rms(self._last(x, lens).float()) @ self.beta

    def run(self, row: dict):
        """Rank order and the exact margin of every doc that reached the last layer.

        Tier keys follow spec5: a later cut ranks above an earlier cut, and the
        survivors rank above every cut, ordered by the frozen margin.
        """
        eng = self.eng
        S = len(self.suffix_ids)
        docs = [c[:-S] for c in row["conts"]]
        n = len(docs)
        cache, P = eng.prefix(row["prefix"], Plan())
        state = {}
        stage_score = torch.zeros(n)
        for idx in chunks([max(1, len(d)) for d in docs], self.budget):
            x, kvs, lens, valid = advance(eng, cache, P, [docs[i] for i in idx], None, None, 0, self.stages[0][0])
            scores = probe_at(eng, self.probe, cache, P, kvs, lens, valid, self.stages[0][0])
            local = {doc_i: t for t, doc_i in enumerate(idx)}
            for doc_i, xi, kvi in slice_state(x, kvs, lens, idx):
                state[doc_i] = {"x": xi, "kvs": kvi}
                stage_score[doc_i] = float(scores[local[doc_i]])
        tier_key = torch.zeros(n)
        tier_score = stage_score.clone()
        alive = keep_top(stage_score, list(range(n)), self.stages[0][1])
        dropped = [i for i in range(n) if i not in set(alive)]
        for i in dropped:
            tier_key[i] = 1
        depth = self.stages[0][0]
        for t, (layer, k) in enumerate(self.stages[1:], start=2):
            alive, stage_score = self._probe_stage(cache, P, docs, state, alive, depth, layer, k, stage_score)
            for i in range(n):
                if tier_key[i] == 0 and i not in set(alive):
                    tier_key[i] = t
                    tier_score[i] = stage_score[i]
            depth = layer
        del state
        return self._finish(cache, P, row["conts"], alive, tier_key, tier_score)

    def _probe_stage(self, cache, P, docs, state, alive, depth, layer, k, stage_score):
        eng = self.eng
        scores = stage_score.clone()
        for idx in chunks([state[alive[i]]["x"].shape[0] for i in range(len(alive))], self.budget):
            chosen = [alive[i] for i in idx]
            batch = [(i, state[i]["x"], state[i]["kvs"]) for i in chosen]
            x, kvs, lens_list = batch_state(batch)
            x, kvs, lens, valid = advance(eng, cache, P, [docs[i] for i in chosen], x, kvs, depth + 1, layer)
            sc = probe_at(eng, self.probe, cache, P, kvs, lens, valid, layer)
            for local, doc_i in enumerate(chosen):
                n = int(lens[local])
                state[doc_i]["x"] = x[local, :n].contiguous()
                state[doc_i]["kvs"] = [(kk[local, :, :n].contiguous(), vv[local, :, :n].contiguous()) for kk, vv in kvs]
                scores[doc_i] = float(sc[local])
        return keep_top(scores, alive, k), scores

    def _finish(self, cache, P, conts, alive, tier_key, tier_score):
        """Ridge and final margin on the joint document+suffix forward.

        A document-only cache matches the probe's training, but in bf16 it does
        not reproduce the joint residual the ridge was fit on. Survivors are
        rescored from the start as one sequence, which matches Manual.score.
        """
        eng = self.eng
        layer = self.ridge_layer
        scores = tier_score.clone()
        state = {}
        for idx in chunks([max(1, len(conts[alive[i]])) for i in range(len(alive))], self.budget):
            chosen = [alive[i] for i in idx]
            x, kvs, lens, valid = advance(eng, cache, P, [conts[i] for i in chosen], None, None, 0, layer)
            sc = self.ridge_of(x, lens)
            for local, doc_i in enumerate(chosen):
                ntok = int(lens[local])
                state[doc_i] = {
                    "x": x[local, :ntok].contiguous(),
                    "kvs": [(kk[local, :, :ntok].contiguous(), vv[local, :, :ntok].contiguous()) for kk, vv in kvs],
                }
                scores[doc_i] = float(sc[local])
        last = keep_top(scores, alive, self.ridge_keep)
        last_set = set(last)
        for i in alive:
            if i not in last_set:
                tier_key[i] = 3
                tier_score[i] = scores[i]
        exact = torch.full((len(conts),), float("-inf"))
        for idx in chunks([state[last[i]]["x"].shape[0] for i in range(len(last))], self.budget):
            chosen = [last[i] for i in idx]
            batch = [(i, state[i]["x"], state[i]["kvs"]) for i in chosen]
            x, kvs, lens_list = batch_state(batch)
            x, kvs, lens, valid = advance(eng, cache, P, [conts[i] for i in chosen], x, kvs, layer + 1, eng.L - 1)
            m = self.margins_of(x, lens)
            for local, doc_i in enumerate(chosen):
                exact[doc_i] = float(m[local])
                tier_key[doc_i] = 4
                tier_score[doc_i] = float(m[local])
        order = sorted(range(len(conts)), key=lambda i: (float(tier_key[i]), float(tier_score[i])), reverse=True)
        return order, exact, tier_key, tier_score


def load_all():
    rule = require_rule()
    eng = load_engine(rule["model"], bf16=True)
    saved = torch.load(DATA / "probes/q4_full.pt", map_location="cpu")
    rows0 = torch.load(DATA / "pilot.pt")
    S = rows0[0]["suffix_len"]
    suffix_ids = rows0[0]["conts"][0][-S:]
    probe = Probe(eng, suffix_ids, saved["args"]["exits"], r=saved["args"]["rank"]).to(eng.device).eval()
    probe.load_state_dict(saved["state"], strict=False)
    beta = ridge_beta()
    return eng, Cascade(eng, probe, beta, suffix_ids), suffix_ids


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a[:10]), set(b[:10])
    inter = len(sa & sb)
    return inter / (20 - inter)


@torch.no_grad()
def check_mode(limit: int) -> None:
    if OK.exists():
        OK.unlink()
    eng, cas, _ = load_all()
    import probe as probe_mod
    from probe import query_scores
    probe_mod.BF16 = True
    saved_heads = torch.load(DATA / "probes/q4_full_check.pt", map_location="cpu")
    by_qid = {r["qid"]: r for r in torch.load(DATA / "check.pt")}
    for qi, qid in enumerate(saved_heads["qids"][:3]):
        row = by_qid[qid]
        official = query_scores(eng, cas.probe, row, 0, 4000)[9].float().cpu()
        ref_h = saved_heads["heads"][9][qi].float().cpu()
        S = len(cas.suffix_ids)
        docs = [c[:-S] for c in row["conts"]]
        cache, P = eng.prefix(row["prefix"], Plan())
        mine = torch.zeros(len(docs))
        for idx in chunks([max(1, len(docs[i])) for i in range(len(docs))], 4000):
            x, kvs, lens, valid = advance(eng, cache, P, [docs[i] for i in idx], None, None, 0, 9)
            sc = probe_at(eng, cas.probe, cache, P, kvs, lens, valid, 9)
            for local, doc_i in enumerate(idx):
                mine[doc_i] = float(sc[local])
        c_off = float(torch.corrcoef(torch.stack([official, ref_h]))[0, 1])
        c_mine = float(torch.corrcoef(torch.stack([mine, official]))[0, 1])
        print(
            f"probe@9 {qid} official-vs-saved {c_off:.4f} "
            f"mine-vs-official {c_mine:.4f} max {float((mine - official).abs().max()):.4f}",
            flush=True,
        )
    rows = torch.load(DATA / "check.pt")
    if limit:
        rows = rows[:limit]
    ref = torch.load(DATA / "runs/check/collect_4b_b8000.pt", map_location="cpu")
    jac, cover = [], []
    qrels, orders, refo = {}, {}, {}
    for n, row in enumerate(rows, 1):
        order, exact, _, _ = cas.run(row)
        margins = ref["queries"][row["qid"]]["margins"]
        ref_order = sorted(range(100), key=lambda i: float(margins[i]), reverse=True)
        jac.append(jaccard(order, ref_order))
        kept = {i for i in range(100) if float(exact[i]) > float("-inf")}
        cover.append(float(set(ref_order[:10]) <= kept))
        orders[row["qid"]] = [row["ids"][i] for i in order]
        refo[row["qid"]] = [row["ids"][i] for i in ref_order]
        if row.get("qrels"):
            qrels[row["qid"]] = row["qrels"]
        if n % 20 == 0 or n == len(rows):
            print(f"check {n}/{len(rows)} running J {sum(jac) / len(jac):.3f}", flush=True)
    report = {
        "n": len(rows),
        "jaccard@10": sum(jac) / len(jac),
        "cover": sum(cover) / len(cover),
        "cascade_ndcg@10": ndcg(qrels, orders),
        "teacher_ndcg@10": ndcg(qrels, refo),
    }
    report["drop"] = report["teacher_ndcg@10"] - report["cascade_ndcg@10"]
    (DATA / "paper_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    # Published R1 on this split is J 0.956 and drop -0.0001. The reuse forward
    # drifts in bf16, so the gate is the ranking, not a bit match.
    if report["jaccard@10"] >= 0.90 and abs(report["drop"]) <= 0.002:
        OK.write_text(json.dumps(report) + "\n")
        print(f"wrote {OK}", flush=True)
    else:
        raise SystemExit("cascade ranking misses the published check result; dev stays unread")


def timed(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


@torch.no_grad()
def time_mode(n: int) -> None:
    if not OK.exists():
        raise SystemExit("run check before time")
    eng, cas, _ = load_all()
    rows = torch.load(DATA / "pilot.pt")[: n + 2]
    for row in rows[:2]:
        cas.run(row)
        score_query(eng, row, Plan(), 4000)
    base, casc = [], []
    for row in rows[2:]:
        base.append(timed(lambda row=row: score_query(eng, row, Plan(), 4000)))
        casc.append(timed(lambda row=row: cas.run(row)))
        print(f"{row['qid']} base {base[-1]:.0f} ms cascade {casc[-1]:.0f} ms", flush=True)
    def pct(xs, p):
        s = sorted(xs)
        return s[min(len(s) - 1, int(round((len(s) - 1) * p)))]
    report = {
        "n": len(base),
        "slice": "pilot[2:n+2], warmup 2",
        "baseline_mean_ms": sum(base) / len(base),
        "cascade_mean_ms": sum(casc) / len(casc),
        "baseline_p95_ms": pct(base, 0.95),
        "cascade_p95_ms": pct(casc, 0.95),
    }
    report["mean_speedup"] = report["baseline_mean_ms"] / report["cascade_mean_ms"]
    dest = DATA / "paper_time.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def build_rows(list_path: Path, qids: list[str]) -> list[dict]:
    from transformers import AutoTokenizer

    manifest = load_manifest()
    pins = manifest["pins"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-4B", padding_side="left")
    prefix_ids = tok.encode(pins["prefix"], add_special_tokens=False)
    suffix_ids = tok.encode(pins["suffix"], add_special_tokens=False)
    wanted = set(qids)
    rows = []
    with list_path.open() as fh:
        for line in fh:
            src = json.loads(line)
            if src["qid"] not in wanted:
                continue
            prefix = query_prefix(tok, pins, prefix_ids, suffix_ids, src["query"])
            hits = src["hits"]
            rows.append({
                "qid": src["qid"],
                "ids": [h["id"] for h in hits],
                "prefix": prefix,
                "conts": [continuation(tok, pins, suffix_ids, h["text"], len(prefix)) for h in hits],
                "suffix_len": len(suffix_ids),
            })
            if len(rows) == len(wanted):
                break
    if len(rows) != len(qids):
        raise SystemExit(f"built {len(rows)} rows, wanted {len(qids)}")
    return rows


def public_metrics(rows, orders, teacher_path: Path, qrels: dict) -> dict:
    teacher = {r["qid"]: r for r in (json.loads(line) for line in teacher_path.read_text().splitlines() if line.strip())}
    jac = []
    rankings = {}
    teacher_rankings = {}
    for row, order in zip(rows, orders):
        ids = [row["ids"][i] for i in order]
        rankings[row["qid"]] = ids
        scores = teacher[row["qid"]]["scores"]
        teacher_rankings[row["qid"]] = sorted(scores, key=lambda d: (float(scores[d]), d), reverse=True)
        jac.append(jaccard(
            [row["ids"].index(d) if d in row["ids"] else -1 for d in ids[:10]],
            [row["ids"].index(d) if d in row["ids"] else -2 for d in teacher_rankings[row["qid"]][:10]],
        ))
    scored = [q for q in rankings if q in qrels]
    return {
        "n": len(rows),
        "jaccard@10_vs_4b": sum(jac) / len(jac),
        "cascade_ndcg@10": ndcg(qrels, {q: rankings[q] for q in scored}),
        "teacher_ndcg@10": ndcg(qrels, {q: teacher_rankings[q] for q in scored}),
    }


@torch.no_grad()
def dev_mode() -> None:
    if not OK.exists():
        raise SystemExit("run check before dev")
    require_rule()
    eng, cas, suffix_ids = load_all()
    manifest = load_manifest()
    qids = [str(q) for q in manifest["arabic_dev_ids"]]
    lists = ROOT / "artifacts/ops/decision_residual/cascade/top100.jsonl"
    dest = DATA / "paper_dev.jsonl"
    done = set()
    if dest.exists():
        for line in dest.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["qid"])
    todo = [q for q in qids if q not in done]
    print(f"dev todo {len(todo)}", flush=True)
    # Stream the lists once into a dict only for the remaining ids.
    wanted = set(todo)
    by_q = {}
    if wanted:
        with lists.open() as fh:
            for line in fh:
                src = json.loads(line)
                if src["qid"] in wanted:
                    by_q[src["qid"]] = src
                    if len(by_q) == len(wanted):
                        break
    from transformers import AutoTokenizer
    pins = manifest["pins"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-4B", padding_side="left")
    prefix_ids = tok.encode(pins["prefix"], add_special_tokens=False)
    suf = tok.encode(pins["suffix"], add_special_tokens=False)
    if suf != suffix_ids:
        raise SystemExit("suffix ids do not match the probe")
    with dest.open("a") as fh:
        for n, qid in enumerate(todo, 1):
            src = by_q[qid]
            prefix = query_prefix(tok, pins, prefix_ids, suf, src["query"])
            row = {
                "qid": qid,
                "ids": [h["id"] for h in src["hits"]],
                "prefix": prefix,
                "conts": [continuation(tok, pins, suf, h["text"], len(prefix)) for h in src["hits"]],
                "suffix_len": len(suf),
            }
            order, _, _, _ = cas.run(row)
            fh.write(json.dumps({"qid": qid, "order": [row["ids"][i] for i in order]}) + "\n")
            if n % 20 == 0:
                fh.flush()
                print(f"dev {n}/{len(todo)}", flush=True)
    import mteb
    task = mteb.get_tasks(tasks=["MIRACLRetrieval"], languages=["ara"])[0]
    task.load_data()
    raw = task.dataset["ar"]["dev"]["relevant_docs"]
    qrels = {str(q): {str(d): int(r) for d, r in docs.items()} for q, docs in raw.items() if str(q) in set(qids) and docs}
    orders = []
    rows = []
    saved = {json.loads(line)["qid"]: json.loads(line)["order"] for line in dest.read_text().splitlines() if line.strip()}
    # Rebuild id lists in manifest order from the dev file.
    idmap = {}
    with lists.open() as fh:
        for line in fh:
            src = json.loads(line)
            if src["qid"] in saved:
                idmap[src["qid"]] = [h["id"] for h in src["hits"]]
    for qid in qids:
        rows.append({"qid": qid, "ids": idmap[qid]})
        orders.append([idmap[qid].index(d) for d in saved[qid]])
    metrics = public_metrics(rows, orders, OUT / "dev_4b.jsonl", qrels)
    (DATA / "paper_dev.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


@torch.no_grad()
def dl19_mode() -> None:
    if not OK.exists():
        raise SystemExit("run check before dl19")
    require_rule()
    eng, cas, suffix_ids = load_all()
    manifest = load_manifest()
    raw_ids = manifest["english_pool"]["TRECDL2019"]
    qids = [str(q["id"] if isinstance(q, dict) else q) for q in raw_ids]
    rows = build_rows(OUT / "english/TRECDL2019_top100.jsonl", qids)
    orders = []
    for n, row in enumerate(rows, 1):
        order, _, _, _ = cas.run(row)
        orders.append(order)
        if n % 10 == 0:
            print(f"dl19 {n}/{len(rows)}", flush=True)
    import mteb
    task = mteb.get_tasks(tasks=["TRECDL2019"])[0]
    task.load_data()
    split = task.dataset["default"]["test"] if "default" in task.dataset else task.dataset["test"]
    raw = split["relevant_docs"]
    qrels = {str(q): {str(d): int(r) for d, r in docs.items()} for q, docs in raw.items()}
    metrics = public_metrics(rows, orders, OUT / "english/TRECDL2019_4b.jsonl", qrels)
    (DATA / "paper_dl19.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


@torch.no_grad()
def unit_mode() -> None:
    """One document, full depth, against Manual.score. Isolates the reuse forward from the cascade."""
    eng, cas, suffix_ids = load_all()
    row = torch.load(DATA / "check.pt")[0]
    S = len(suffix_ids)
    cache, P = eng.prefix(row["prefix"], Plan())
    i = 8
    doc = row["conts"][i][:-S]
    n = len(doc)
    xj0, _, _, _ = advance(eng, cache, P, [row["conts"][i]], None, None, 0, 0)
    xd0, kvsd0, lensd, _ = advance(eng, cache, P, [doc], None, None, 0, 0)
    print(
        f"layer0 docpos0 {float((xd0[0, 0] - xj0[0, 0]).abs().max()):.5f} "
        f"docposLast {float((xd0[0, -1] - xj0[0, n - 1]).abs().max()):.5f}",
        flush=True,
    )
    sx = suffix_through(eng, cache, P, kvsd0, lensd, suffix_ids, 0, 0, None)
    print(f"suffix layer0 vs joint {float((sx[0] - xj0[0, n:]).abs().max()):.5f}", flush=True)
    W = n + S
    causal = torch.ones((W, W), dtype=torch.bool, device=eng.device).tril()
    print(f"P {P} n {n} S {S} query0_true {int(causal[0].sum()) + P} query_last_doc_true {int(causal[n - 1].sum()) + P}", flush=True)


@torch.no_grad()
def profile_mode(n: int) -> None:
    """Per-stage wall-clock of the timed cascade on pilot queries, plus one real query's tiers."""
    eng, cas, _ = load_all()
    g = globals()
    spent: dict[str, float] = {}
    tokens: dict[str, int] = {}

    def wrap(name, fn, key):
        def inner(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            torch.cuda.synchronize()
            k = key(a)
            spent[k] = spent.get(k, 0.0) + (time.perf_counter() - t0) * 1000
            return out
        return inner

    def adv_key(a):
        rows = a[3]
        k = f"layers {a[6]}-{a[7]}" + (" joint" if len(rows[0]) and rows[0][-len(cas.suffix_ids):] == cas.suffix_ids else "")
        tokens[k] = tokens.get(k, 0) + sum(len(r) for r in rows)
        return k

    g["advance"] = wrap("advance", advance, adv_key)
    g["probe_at"] = wrap("probe_at", probe_at, lambda a: f"probe to {a[7]}")
    g["batch_state"] = wrap("batch_state", batch_state, lambda a: "re-pad cache")
    g["slice_state"] = wrap("slice_state", slice_state, lambda a: "unpad cache")
    rows = torch.load(DATA / "pilot.pt")[: n + 2]
    for row in rows[:2]:
        cas.run(row)
        score_query(eng, row, Plan(), 4000)
    spent.clear()
    tokens.clear()
    total, base = 0.0, 0.0
    for row in rows[2:]:
        total += timed(lambda row=row: cas.run(row))
        base += timed(lambda row=row: score_query(eng, row, Plan(), 4000))
    q = len(rows) - 2
    out = {
        "n": q,
        "baseline_ms": base / q,
        "cascade_ms": total / q,
        "parts_ms": {k: v / q for k, v in sorted(spent.items(), key=lambda kv: -kv[1])},
        "tokens_per_query": {k: v / q for k, v in tokens.items()},
    }
    out["other_ms"] = out["cascade_ms"] - sum(out["parts_ms"].values())
    row = rows[2]
    order, exact, tier_key, tier_score = cas.run(row)
    S = len(cas.suffix_ids)
    out["example"] = {
        "qid": row["qid"],
        "prefix_tokens": len(row["prefix"]),
        "doc_tokens": [len(c) - S for c in row["conts"]],
        "suffix_tokens": S,
        "tier": [int(t) for t in tier_key.tolist()],
    }
    (DATA / "paper_profile.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "example"}, indent=2), flush=True)


def main() -> None:
    mode = sys.argv[1]
    if mode == "freeze":
        freeze_rule()
    elif mode == "check":
        check_mode(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif mode == "time":
        time_mode(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif mode == "dev":
        dev_mode()
    elif mode == "dl19":
        dl19_mode()
    elif mode == "unit":
        unit_mode()
    elif mode == "profile":
        profile_mode(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    else:
        raise SystemExit("mode: freeze|check|time|dev|dl19")


if __name__ == "__main__":
    main()
