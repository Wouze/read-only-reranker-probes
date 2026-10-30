#!/usr/bin/env python3
"""Week queue: joint suffix (A), short probes are trained by probe.py (B, C),
document-prefix cut (D), multi-token ridge (E).

Pilot, fit, and check only. The 261-query holdout is not opened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from core import Plan, score_query  # noqa: E402
from fit_rules import ndcg  # noqa: E402
from next_exps import _rank_report  # noqa: E402
from paper_eval import DATA, advance, batch_state, chunks, jaccard, keep_top, load_all, ridge_beta, timed  # noqa: E402
from spec import fit_ridge, rms  # noqa: E402

LAYERS_E = (15, 17, 19, 21)


def probe_seeing(eng, probe, cache, P, kvs, doc_lens, layer: int):
    """Probe at `layer`. It sits just after the document and cannot see later tokens."""
    B = doc_lens.shape[0]
    S = probe.emb.shape[0]
    hq, hd = eng.hq, eng.hd
    x = probe.emb[None].expand(B, S, -1).clone()
    pos = P + doc_lens[:, None] + torch.arange(S, device=doc_lens.device)
    cos, sin = eng.base.rotary_emb(x, pos)
    W = kvs[0][0].shape[2]
    visible = torch.arange(W, device=doc_lens.device)[None] < doc_lens[:, None]
    mask = torch.cat([
        torch.ones((B, S, P), dtype=torch.bool, device=doc_lens.device),
        visible[:, None, :].expand(B, S, W),
        torch.ones((S, S), dtype=torch.bool, device=doc_lens.device).tril()[None].expand(B, S, S),
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


def tail_through(eng, cache, P, prefix_kvs, prefix_lens, tails: list[list[int]], end: int):
    """Run each document's tail through layers 0..end, reading the prefix cache."""
    dev = eng.device
    B = len(tails)
    W = max(1, max(len(t) for t in tails))
    ids = torch.zeros((B, W), dtype=torch.long, device=dev)
    lens = torch.tensor([len(t) for t in tails], device=dev)
    for i, t in enumerate(tails):
        if t:
            ids[i, : len(t)] = torch.tensor(t, device=dev)
    x = eng.base.embed_tokens(ids)
    pos = P + prefix_lens[:, None] + torch.arange(W, device=dev)
    cos, sin = eng.base.rotary_emb(x, pos)
    valid = torch.arange(W, device=dev)[None] < lens[:, None]
    causal = torch.ones((W, W), dtype=torch.bool, device=dev).tril()[None] & valid[:, None, :]
    Wp = prefix_kvs[0][0].shape[2]
    pre_ok = torch.arange(Wp, device=dev)[None] < prefix_lens[:, None]
    mask = torch.cat([
        torch.ones((B, W, P), dtype=torch.bool, device=dev),
        pre_ok[:, None, :].expand(B, W, Wp),
        causal,
    ], 2)
    import paper_eval as pe
    from core import Plan
    plan = Plan()
    skvs = []
    for j in range(end + 1):
        layer = eng.base.layers[j]
        if pe.COMPACT == "all":
            q, k, v = pe.proj_real(eng, layer, x, valid, plan)
        else:
            q, k, v = eng._proj(layer, layer.input_layernorm(x), plan)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        skvs.append((k, v))
        pk, pv = cache[j]
        dk, dv = prefix_kvs[j]
        kf = torch.cat([pk.expand(B, -1, -1, -1), dk, k], 2)
        vf = torch.cat([pv.expand(B, -1, -1, -1), dv, v], 2)
        a = F.scaled_dot_product_attention(q, kf, vf, attn_mask=mask[:, None], enable_gqa=True, scale=layer.self_attn.scaling)
        joined = a.transpose(1, 2).reshape(B, W, -1)
        if pe.COMPACT == "all":
            x = x + pe.apply_real(joined, valid, layer.self_attn.o_proj)
        else:
            x = x + layer.self_attn.o_proj(joined)
        if pe.COMPACT in ("mlp", "all"):
            x = x + pe.apply_real(x, valid, lambda h, layer=layer: layer.mlp(layer.post_attention_layernorm(h)))
        else:
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = x * valid[..., None]
    return x, skvs, lens


def _stitch(prefix_x, prefix_kvs, prefix_lens, tail_x, tail_kvs, tail_lens):
    """Concatenate a document prefix and its tail along the sequence."""
    B = prefix_x.shape[0]
    d = prefix_x.shape[-1]
    lens = [int(prefix_lens[i]) + int(tail_lens[i]) for i in range(B)]
    W = max(lens)
    x = prefix_x.new_zeros((B, W, d))
    hkv, hd = prefix_kvs[0][0].shape[1], prefix_kvs[0][0].shape[-1]
    kvs = []
    for _j in range(len(prefix_kvs)):
        kvs.append((prefix_x.new_zeros(B, hkv, W, hd), prefix_x.new_zeros(B, hkv, W, hd)))
    for i in range(B):
        n = int(prefix_lens[i])
        m = int(tail_lens[i])
        x[i, :n] = prefix_x[i, :n]
        if m:
            x[i, n : n + m] = tail_x[i, :m]
        for j in range(len(prefix_kvs)):
            kvs[j][0][i, :, :n] = prefix_kvs[j][0][i, :, :n]
            kvs[j][1][i, :, :n] = prefix_kvs[j][1][i, :, :n]
            if m:
                kvs[j][0][i, :, n : n + m] = tail_kvs[j][0][i, :, :m]
                kvs[j][1][i, :, n : n + m] = tail_kvs[j][1][i, :, :m]
    return x, kvs, lens


class JointSuffix:
    """Document and suffix run together from layer 0. Probes see only the document slice."""

    def __init__(self, eng, probe, beta, suffix_ids, budget=4000):
        self.eng = eng
        self.probe = probe
        self.beta = beta.to(eng.device)
        self.suffix_ids = suffix_ids
        self.budget = budget
        self.S = len(suffix_ids)

    def _last(self, x, lens):
        rows = torch.arange(x.shape[0], device=x.device)
        return x[rows, torch.tensor(lens, device=x.device) - 1]

    def _ridge(self, x, lens):
        return rms(self._last(x, lens).float()) @ self.beta

    def _margin(self, x, lens):
        h = self.eng.base.norm(self._last(x, lens)).float()
        return h @ self.eng.w

    def run(self, row):
        eng = self.eng
        S = self.S
        conts = row["conts"]
        n = len(conts)
        doc_lens_all = [len(c) - S for c in conts]
        cache, P = eng.prefix(row["prefix"], Plan())
        state = {}
        stage = torch.zeros(n)
        for idx in chunks([max(1, len(conts[i])) for i in range(n)], self.budget):
            x, kvs, lens, _v = advance(eng, cache, P, [conts[i] for i in idx], None, None, 0, 9)
            dl = torch.tensor([doc_lens_all[i] for i in idx], device=eng.device)
            sc = probe_seeing(eng, self.probe, cache, P, kvs, dl, 9)
            for local, doc_i in enumerate(idx):
                ntok = int(lens[local])
                state[doc_i] = {
                    "x": x[local, :ntok].contiguous(),
                    "kvs": [(k[local, :, :ntok].contiguous(), v[local, :, :ntok].contiguous()) for k, v in kvs],
                }
                stage[doc_i] = float(sc[local])
        tier_key = torch.zeros(n)
        tier_score = stage.clone()
        alive = keep_top(stage, list(range(n)), 60)
        for i in range(n):
            if i not in set(alive):
                tier_key[i] = 1
        alive, stage = self._continue_probe(cache, P, conts, doc_lens_all, state, alive, 9, 15, 40)
        for i in range(n):
            if tier_key[i] == 0 and i not in set(alive):
                tier_key[i] = 2
                tier_score[i] = stage[i]
        return self._finish(cache, P, conts, state, alive, tier_key, tier_score)

    def _continue_probe(self, cache, P, conts, doc_lens_all, state, alive, depth, layer, k):
        eng = self.eng
        scores = torch.zeros(len(conts))
        for idx in chunks([state[alive[i]]["x"].shape[0] for i in range(len(alive))], self.budget):
            chosen = [alive[i] for i in idx]
            x, kvs, lens_list = batch_state([(i, state[i]["x"], state[i]["kvs"]) for i in chosen])
            x, kvs, lens, _v = advance(eng, cache, P, [conts[i] for i in chosen], x, kvs, depth + 1, layer)
            dl = torch.tensor([doc_lens_all[i] for i in chosen], device=eng.device)
            sc = probe_seeing(eng, self.probe, cache, P, kvs, dl, layer)
            for local, doc_i in enumerate(chosen):
                ntok = int(lens[local])
                state[doc_i] = {
                    "x": x[local, :ntok].contiguous(),
                    "kvs": [(kk[local, :, :ntok].contiguous(), vv[local, :, :ntok].contiguous()) for kk, vv in kvs],
                }
                scores[doc_i] = float(sc[local])
        return keep_top(scores, alive, k), scores

    def _finish(self, cache, P, conts, state, alive, tier_key, tier_score):
        eng = self.eng
        scores = tier_score.clone()
        if not alive:
            order = sorted(range(len(conts)), key=lambda i: (float(tier_key[i]), float(tier_score[i])), reverse=True)
            return order, torch.full((len(conts),), float("-inf"))
        for idx in chunks([state[alive[i]]["x"].shape[0] for i in range(len(alive))], self.budget):
            chosen = [alive[i] for i in idx]
            x, kvs, lens_list = batch_state([(i, state[i]["x"], state[i]["kvs"]) for i in chosen])
            x, kvs, lens, _v = advance(eng, cache, P, [conts[i] for i in chosen], x, kvs, 16, 21)
            sc = self._ridge(x, lens.tolist())
            for local, doc_i in enumerate(chosen):
                ntok = int(lens[local])
                state[doc_i] = {
                    "x": x[local, :ntok].contiguous(),
                    "kvs": [(kk[local, :, :ntok].contiguous(), vv[local, :, :ntok].contiguous()) for kk, vv in kvs],
                }
                scores[doc_i] = float(sc[local])
        last = keep_top(scores, alive, 15)
        last_set = set(last)
        for i in alive:
            if i not in last_set:
                tier_key[i] = 3
                tier_score[i] = scores[i]
        exact = torch.full((len(conts),), float("-inf"))
        for idx in chunks([state[last[i]]["x"].shape[0] for i in range(len(last))], self.budget):
            chosen = [last[i] for i in idx]
            x, kvs, _ll = batch_state([(i, state[i]["x"], state[i]["kvs"]) for i in chosen])
            x, kvs, lens, _v = advance(eng, cache, P, [conts[i] for i in chosen], x, kvs, 22, eng.L - 1)
            got = self._margin(x, lens.tolist())
            for local, doc_i in enumerate(chosen):
                exact[doc_i] = float(got[local])
                tier_key[doc_i] = 4
                tier_score[doc_i] = float(got[local])
        order = sorted(range(len(conts)), key=lambda i: (float(tier_key[i]), float(tier_score[i])), reverse=True)
        return order, exact


class PrefixCut(JointSuffix):
    """Layer-9 probe sees only the first M document tokens. Survivors catch up the tail."""

    def __init__(self, eng, probe, beta, suffix_ids, limit_m: int, budget=4000):
        super().__init__(eng, probe, beta, suffix_ids, budget)
        self.M = limit_m

    def run(self, row):
        eng = self.eng
        S = self.S
        M = self.M
        conts = row["conts"]
        docs = [c[:-S] for c in conts]
        n = len(docs)
        cache, P = eng.prefix(row["prefix"], Plan())
        prefixes = [d[:M] for d in docs]
        state = {}
        stage = torch.zeros(n)
        for idx in chunks([max(1, len(prefixes[i])) for i in range(n)], self.budget):
            x, kvs, lens, _v = advance(eng, cache, P, [prefixes[i] for i in idx], None, None, 0, 9)
            dl = lens.clone()
            sc = probe_seeing(eng, self.probe, cache, P, kvs, dl, 9)
            for local, doc_i in enumerate(idx):
                ntok = int(lens[local])
                state[doc_i] = {
                    "x": x[local, :ntok].contiguous(),
                    "kvs": [(k[local, :, :ntok].contiguous(), v[local, :, :ntok].contiguous()) for k, v in kvs],
                    "n": ntok,
                }
                stage[doc_i] = float(sc[local])
        tier_key = torch.zeros(n)
        tier_score = stage.clone()
        alive = keep_top(stage, list(range(n)), 60)
        for i in range(n):
            if i not in set(alive):
                tier_key[i] = 1
        # Catch the tail up through layer 9, then join the suffix and continue.
        full = {}
        for idx in chunks([max(1, len(docs[alive[i]])) for i in range(len(alive))], self.budget):
            chosen = [alive[i] for i in idx]
            tails = [docs[i][state[i]["n"]:] for i in chosen]
            if any(tails):
                px, pkvs, plens = batch_state([(i, state[i]["x"], state[i]["kvs"]) for i in chosen])
                pre_lens = torch.tensor([state[i]["n"] for i in chosen], device=eng.device)
                tx, tkvs, tlens = tail_through(eng, cache, P, pkvs, pre_lens, tails, 9)
                xj, kvsj, lensj = _stitch(px, pkvs, pre_lens, tx, tkvs, tlens)
            else:
                xj, kvsj, lensj = batch_state([(i, state[i]["x"], state[i]["kvs"]) for i in chosen])
            # Append the suffix by a joint continuation from layer 10 on doc+suffix.
            # The suffix was not in layers 0-9; run it against the caught-up document.
            from paper_eval import suffix_through
            doc_lens = torch.tensor(lensj if isinstance(lensj, list) else lensj.tolist(), device=eng.device)
            sx = suffix_through(eng, cache, P, kvsj, doc_lens, self.suffix_ids, 0, 9, None)
            B = len(chosen)
            width = max(int(doc_lens[i]) + S for i in range(B))
            xjs = xj.new_zeros((B, width, xj.shape[-1]))
            dummy_k = []
            hkv, hd = kvsj[0][0].shape[1], kvsj[0][0].shape[-1]
            for j in range(len(kvsj)):
                kk = xj.new_zeros(B, hkv, width, hd)
                vv = xj.new_zeros(B, hkv, width, hd)
                for i in range(B):
                    ntok = int(doc_lens[i])
                    xjs[i, :ntok] = xj[i, :ntok]
                    xjs[i, ntok : ntok + S] = sx[i]
                    kk[i, :, :ntok] = kvsj[j][0][i, :, :ntok]
                    vv[i, :, :ntok] = kvsj[j][1][i, :, :ntok]
                dummy_k.append((kk, vv))
            tokens = [conts[i] for i in chosen]
            xjs, kvs2, lens, _v = advance(eng, cache, P, tokens, xjs, dummy_k, 10, 15)
            dl = doc_lens
            sc = probe_seeing(eng, self.probe, cache, P, kvs2, dl, 15)
            for local, doc_i in enumerate(chosen):
                ntok = int(lens[local])
                full[doc_i] = {
                    "x": xjs[local, :ntok].contiguous(),
                    "kvs": [(kk[local, :, :ntok].contiguous(), vv[local, :, :ntok].contiguous()) for kk, vv in kvs2],
                }
                stage[doc_i] = float(sc[local])
        alive2 = keep_top(stage, alive, 40)
        for i in alive:
            if i not in set(alive2):
                tier_key[i] = 2
                tier_score[i] = stage[i]
        return self._finish(cache, P, conts, full, alive2, tier_key, tier_score)


def _time_and_score(tag: str, builder, limit: int, time_n: int) -> None:
    eng, cas, _s = load_all()
    model = builder(eng, cas)
    if time_n:
        rows = torch.load(DATA / "pilot.pt")[: time_n + 2]
        for row in rows[:2]:
            model.run(row)
            score_query(eng, row, Plan(), 4000)
        base, ours = [], []
        for row in rows[2:]:
            base.append(timed(lambda row=row: score_query(eng, row, Plan(), 4000)))
            ours.append(timed(lambda row=row: model.run(row)))
            print(f"{tag} {row['qid']} base {base[-1]:.0f} ours {ours[-1]:.0f}", flush=True)
        report = {
            "n": len(base),
            "baseline_mean_ms": sum(base) / len(base),
            "method_mean_ms": sum(ours) / len(ours),
        }
        report["mean_speedup"] = report["baseline_mean_ms"] / report["method_mean_ms"]
        (DATA / f"week_{tag}_time.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    rows = torch.load(DATA / "check.pt")
    if limit:
        rows = rows[:limit]
    orders, exacts = [], []
    for n, row in enumerate(rows, 1):
        order, exact = model.run(row)
        orders.append(order)
        exacts.append(exact)
        if n % 20 == 0 or n == len(rows):
            print(f"{tag} {n}/{len(rows)}", flush=True)
    report = _rank_report(rows, orders, exacts)
    report["tag"] = tag
    (DATA / f"week_{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


@torch.no_grad()
def joint_mode(limit: int, time_n: int) -> None:
    def build(eng, cas):
        return JointSuffix(eng, cas.probe, cas.beta, cas.suffix_ids, cas.budget)
    _time_and_score("joint", build, limit, time_n)


@torch.no_grad()
def prefix_mode(limit_m: int, limit: int, time_n: int) -> None:
    def build(eng, cas):
        return PrefixCut(eng, cas.probe, cas.beta, cas.suffix_ids, limit_m, cas.budget)
    _time_and_score(f"prefix{limit_m}", build, limit, time_n)


@torch.no_grad()
def collect_suffix(split: str) -> None:
    """Memmap of the 9 suffix hidden states at layers 15, 17, 19, 21. Joint forward."""
    eng, cas, suffix_ids = load_all()
    rows = torch.load(DATA / f"{split}.pt")
    if split == "fit":
        teacher = torch.load(DATA / "runs/fit/collect_4b_b8000.pt", map_location="cpu")["queries"]
        rows = [r for r in rows if r["qid"] in teacher]
    S = len(suffix_ids)
    path = DATA / f"week_suffix_{split}.npy"
    qids = [r["qid"] for r in rows]
    (DATA / f"week_suffix_{split}_qids.json").write_text(json.dumps(qids))
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(rows), 100, S, len(LAYERS_E), eng.base.embed_tokens.embedding_dim))
    for qi, row in enumerate(rows):
        cache, P = eng.prefix(row["prefix"], Plan())
        conts = row["conts"]
        acc = np.zeros((100, S, len(LAYERS_E), mm.shape[-1]), dtype=np.float16)
        filled = np.zeros(100, dtype=bool)
        x = kvs = None
        for idx in chunks([max(1, len(c)) for c in conts], 4000):
            x, kvs, lens, _v = advance(eng, cache, P, [conts[i] for i in idx], None, None, 0, LAYERS_E[0])
            stored = {LAYERS_E[0]: x}
            prev = LAYERS_E[0]
            for layer in LAYERS_E[1:]:
                x, kvs, lens, _v = advance(eng, cache, P, [conts[i] for i in idx], x, kvs, prev + 1, layer)
                stored[layer] = x
                prev = layer
            for local, doc_i in enumerate(idx):
                ntok = int(lens[local])
                for li, layer in enumerate(LAYERS_E):
                    acc[doc_i, :, li] = stored[layer][local, ntok - S : ntok].detach().float().cpu().numpy()
                filled[doc_i] = True
        if not filled.all():
            raise RuntimeError(f"{split} query {row['qid']} missed documents")
        mm[qi] = acc
        if qi % 20 == 0 or qi + 1 == len(rows):
            mm.flush()
            print(f"collect {split} {qi + 1}/{len(rows)}", flush=True)
        del x, kvs
    mm.flush()
    print(f"wrote {path}", flush=True)


def _fit_feature(X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """X (Q,100,d), y (Q,100). Returns beta, held-out corr, lam."""
    cut = int(0.85 * X.shape[0])
    best = None
    for lam in (1e-4, 1e-3, 1e-2):
        beta = fit_ridge(X[:cut], y[:cut], lam)
        pred = X[cut:] @ beta
        c = float(np.nanmean([np.corrcoef(pred[i].numpy(), y[cut + i].numpy())[0, 1] for i in range(len(pred))]))
        if best is None or c > best[0]:
            best = (c, lam)
    beta = fit_ridge(X, y, best[1])
    return beta, best[0], best[1]


def multi_ridge() -> None:
    import spec5

    spec5.set_model("4b")
    fit_q = json.loads((DATA / "week_suffix_fit_qids.json").read_text())
    check_q = json.loads((DATA / "week_suffix_check_qids.json").read_text())
    fit_mm = np.load(DATA / "week_suffix_fit.npy", mmap_mode="r")
    check_mm = np.load(DATA / "week_suffix_check.npy", mmap_mode="r")
    frun = torch.load(DATA / "runs/fit/collect_4b_b8000.pt", map_location="cpu")
    y = torch.tensor([frun["queries"][q]["margins"] for q in fit_q], dtype=torch.float32)
    heads = torch.load(DATA / "probes/q4_full_check.pt", map_location="cpu")
    by_q = {q: i for i, q in enumerate(heads["qids"])}
    sp = spec5.Split("check")
    beta21 = ridge_beta()
    layer_index = {layer: i for i, layer in enumerate(LAYERS_E)}

    def feat(mm, which: str, layer: int):
        sl = mm[:, :, :, layer_index[layer], :].astype(np.float32)
        if which == "last":
            raw = sl[:, :, -1, :]
        elif which == "last4":
            raw = sl[:, :, -4:, :].reshape(sl.shape[0], sl.shape[1], -1)
        elif which == "all9":
            raw = sl.reshape(sl.shape[0], sl.shape[1], -1)
        else:
            raise RuntimeError(which)
        return torch.from_numpy(np.ascontiguousarray(raw))

    specs = [("last", 15), ("last", 17), ("last", 19), ("last", 21), ("last4", 15), ("last4", 19), ("all9", 15), ("all9", 19)]
    report = {}
    h9 = torch.stack([heads["heads"][9][by_q[q]] for q in sp.qids]).numpy()
    h15 = torch.stack([heads["heads"][15][by_q[q]] for q in sp.qids]).numpy()
    check_pos = {q: i for i, q in enumerate(check_q)}
    order_idx = [check_pos[q] for q in sp.qids]
    for which, layer in specs:
        print(f"fit {which}@{layer}", flush=True)
        Xf = feat(fit_mm, which, layer)
        Xf = torch.stack([rms(Xf[i].float()) for i in range(Xf.shape[0])])
        beta, corr, lam = _fit_feature(Xf, y)
        Xc = feat(check_mm, which, layer)
        Xc = torch.stack([rms(Xc[i].float()) for i in order_idx])
        scores = (Xc @ beta).numpy()
        mac = spec5.run(sp, [
            ("probe", 0, 9, h9, 60, "q4"),
            ("probe", 0, 15, h15, 40, "q4"),
            ("full", 0, layer, scores, 15, "ridge"),
        ])
        report[f"{which}@{layer}"] = {
            "lam": lam,
            "heldout_corr": corr,
            "mac_frac": mac["mac_frac"],
            "drop": mac["drop"],
            "jaccard@10": mac["jaccard@10"],
            "cover": mac["cover"],
            "spec5_ndcg@10": mac["ndcg@10"],
        }
        print(json.dumps(report[f"{which}@{layer}"]), flush=True)
        del Xf, Xc, beta
        (DATA / "week_multi_ridge.json").write_text(json.dumps(report, indent=2) + "\n")
    # Sanity: the saved layer-21 last-token ridge, same accountant.
    k = sp.full["layers"].index(21)
    X = torch.stack([rms(sp.full["hidden"][q][:, k].float()) for q in sp.qids])
    base_scores = (X @ beta21).numpy()
    mac = spec5.run(sp, [
        ("probe", 0, 9, h9, 60, "q4"),
        ("probe", 0, 15, h15, 40, "q4"),
        ("full", 0, 21, base_scores, 15, "ridge"),
    ])
    report["saved_last@21"] = {"mac_frac": mac["mac_frac"], "drop": mac["drop"], "jaccard@10": mac["jaccard@10"], "cover": mac["cover"]}
    (DATA / "week_multi_ridge.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def eval_probe(tag: str, second: str | None = None) -> None:
    """Cascade a trained probe's check scores with the layer-21 ridge."""
    import spec5

    spec5.set_model("4b")
    sp = spec5.Split("check")
    beta = ridge_beta()
    k = sp.full["layers"].index(21)
    X = torch.stack([rms(sp.full["hidden"][q][:, k].float()) for q in sp.qids])
    ridge = (X @ beta).numpy()

    def load_heads(name):
        blob = torch.load(DATA / "probes" / f"{name}_check.pt", map_location="cpu")
        pos = {q: i for i, q in enumerate(blob["qids"])}
        return blob, pos

    first, pos = load_heads(tag)
    exits = sorted(int(e) for e in first["heads"])
    if second:
        other, pos2 = load_heads(second)
        h_early = torch.stack([first["heads"][exits[0]][pos[q]] for q in sp.qids]).numpy()
        late_exits = sorted(int(e) for e in other["heads"])
        h_late = torch.stack([other["heads"][late_exits[0]][pos2[q]] for q in sp.qids]).numpy()
        early_layer, late_layer = exits[0], late_exits[0]
    else:
        h_early = torch.stack([first["heads"][9][pos[q]] for q in sp.qids]).numpy()
        h_late = torch.stack([first["heads"][15][pos[q]] for q in sp.qids]).numpy()
        early_layer, late_layer = 9, 15
    mac = spec5.run(sp, [
        ("probe", 0, early_layer, h_early, 60, tag),
        ("probe", 0, late_layer, h_late, 40, second or tag),
        ("full", 0, 21, ridge, 15, "ridge"),
    ])
    report = {"tag": tag, "second": second, "early": early_layer, "late": late_layer, **{k: mac[k] for k in ("mac_frac", "drop", "jaccard@10", "cover", "ndcg@10")}}
    dest = DATA / f"week_eval_{tag}{'_' + second if second else ''}.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def load_saved_probe(eng, suffix_ids, tag: str):
    from probe import Probe

    state = torch.load(DATA / "probes" / f"{tag}.pt", map_location=eng.device)
    args = state["args"]
    n = int(args.get("probe_len") or 0)
    ids = list(suffix_ids)
    if n:
        ids = ids[-n:]
    probe = Probe(eng, ids, [int(e) for e in args["exits"]], r=int(args["rank"])).to(eng.device)
    probe.start_layer = int(args.get("probe_start", 0) or 0)
    missing, unexpected = probe.load_state_dict(state["state"], strict=False)
    bad = [k for k in missing if not k.endswith(".base")]
    if bad or unexpected:
        raise RuntimeError(f"{tag} missing {bad[:8]} unexpected {unexpected[:8]}")
    print(f"loaded {tag} tokens {len(ids)} exits {probe.exits}", flush=True)
    return probe


@torch.no_grad()
def p1_catchup(limit: int, time_n: int) -> None:
    """1-token probe inside the suffix catch-up. Does not touch paper_catchup*.json."""
    from next_exps import Catchup

    eng, cas, suffix_ids = load_all()
    probe = load_saved_probe(eng, suffix_ids, "q4_p1")
    catch = Catchup(eng, probe, cas.beta, suffix_ids, cas.budget)
    if time_n:
        rows = torch.load(DATA / "pilot.pt")[: time_n + 2]
        for row in rows[:2]:
            catch.run(row)
            score_query(eng, row, Plan(), 4000)
        base, ours = [], []
        for row in rows[2:]:
            base.append(timed(lambda row=row: score_query(eng, row, Plan(), 4000)))
            ours.append(timed(lambda row=row: catch.run(row)))
            print(f"p1 {row['qid']} base {base[-1]:.0f} catch {ours[-1]:.0f}", flush=True)
        report = {
            "n": len(base),
            "probe": "q4_p1",
            "schedule": "suffix catch-up, layer-15 head continues the layer-9 residual",
            "baseline_mean_ms": sum(base) / len(base),
            "method_mean_ms": sum(ours) / len(ours),
        }
        report["mean_speedup"] = report["baseline_mean_ms"] / report["method_mean_ms"]
        (DATA / "week_p1_catchup_time.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    rows = torch.load(DATA / "check.pt")
    if limit:
        rows = rows[:limit]
    orders, exacts = [], []
    for n, row in enumerate(rows, 1):
        order, exact, _k, _s = catch.run(row)
        orders.append(order)
        exacts.append(exact)
        if n % 20 == 0 or n == len(rows):
            print(f"p1 catchup {n}/{len(rows)}", flush=True)
    report = _rank_report(rows, orders, exacts)
    report["tag"] = "q4_p1_catchup"
    (DATA / "week_p1_catchup.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    mode = sys.argv[1]
    if mode == "p1catch":
        p1_catchup(int(sys.argv[2]) if len(sys.argv) > 2 else 400, int(sys.argv[3]) if len(sys.argv) > 3 else 30)
        return
    if mode == "joint":
        joint_mode(int(sys.argv[2]) if len(sys.argv) > 2 else 400, int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    elif mode == "prefix":
        prefix_mode(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 400, int(sys.argv[4]) if len(sys.argv) > 4 else 10)
    elif mode == "collect":
        collect_suffix(sys.argv[2])
    elif mode == "multi":
        multi_ridge()
    elif mode == "eval":
        eval_probe(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
