#!/usr/bin/env python3
"""Follow-up measurements for the frozen 4B cascade.

Does not open the 261-query holdout. Does not refit the hashed rule.
Check and pilot only, except a per-query dump of the already-run DL19 lists
so a paired bootstrap can be computed from the same rule.
"""

from __future__ import annotations

import json
import sys
import time
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
from paper_eval import (  # noqa: E402
    DATA,
    Cascade,
    advance,
    batch_state,
    chunks,
    jaccard,
    keep_top,
    load_all,
    probe_at,
    suffix_through,
    timed,
)
from probe import load_engine  # noqa: E402
from train_qwen_draft import OUT, load_manifest  # noqa: E402

MODEL = "Qwen/Qwen3-Reranker-4B"


def _tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def _add(laps, name, t0):
    if laps is None:
        return None
    torch.cuda.synchronize()
    laps[name] = laps.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0
    return time.perf_counter()


def _margin(eng, last: torch.Tensor) -> torch.Tensor:
    h = eng.base.norm(last).float()
    return h @ eng.w


class Catchup(Cascade):
    """Probe on the document, then run only the suffix against that cache.

    The forty survivors are not recomputed from layer 0. From layer 16 on,
    document and suffix residuals continue as one sequence.
    """

    def run(self, row: dict):
        eng = self.eng
        laps = getattr(self, "laps", None)
        t = _tick() if laps is not None else None
        S = len(self.suffix_ids)
        docs = [c[:-S] for c in row["conts"]]
        n = len(docs)
        cache, P = eng.prefix(row["prefix"], Plan())
        t = _add(laps, "prefix", t)
        self._probe_x = {}
        state = {}
        stage_score = torch.zeros(n)
        for idx in chunks([max(1, len(d)) for d in docs], self.budget):
            x, kvs, lens, valid = advance(eng, cache, P, [docs[i] for i in idx], None, None, 0, self.stages[0][0])
            scores, px = probe_forward(eng, self.probe, cache, P, kvs, lens, valid, self.stages[0][0], None, 0)
            local = {doc_i: t for t, doc_i in enumerate(idx)}
            for doc_i, xi, kvi in slice_rows(x, kvs, lens, idx):
                state[doc_i] = {"x": xi, "kvs": kvi}
                self._probe_x[doc_i] = px[local[doc_i]].detach()
                stage_score[doc_i] = float(scores[local[doc_i]])
        t = _add(laps, "layers_0_9_and_probe9", t)
        tier_key = torch.zeros(n)
        tier_score = stage_score.clone()
        alive = keep_top(stage_score, list(range(n)), self.stages[0][1])
        for i in range(n):
            if i not in set(alive):
                tier_key[i] = 1
                self._probe_x.pop(i, None)
        depth = self.stages[0][0]
        for tier_i, (layer, k) in enumerate(self.stages[1:], start=2):
            alive, stage_score = ContCascade._probe_stage(
                self, cache, P, docs, state, alive, depth, layer, k, stage_score
            )
            for i in range(n):
                if tier_key[i] == 0 and i not in set(alive):
                    tier_key[i] = tier_i
                    tier_score[i] = stage_score[i]
            depth = layer
        t = _add(laps, "layers_10_15_and_probe15", t)
        return self._finish(cache, P, row["conts"], alive, tier_key, tier_score, state)

    def _finish(self, cache, P, conts, alive, tier_key, tier_score, state):
        eng = self.eng
        S = len(self.suffix_ids)
        layer = self.ridge_layer
        laps = getattr(self, "laps", None)
        t = _tick() if laps is not None else None
        scores = tier_score.clone()
        joint = {}
        if not alive:
            order = sorted(range(len(conts)), key=lambda i: (float(tier_key[i]), float(tier_score[i])), reverse=True)
            return order, torch.full((len(conts),), float("-inf")), tier_key, tier_score
        for idx in chunks([state[alive[i]]["x"].shape[0] + S for i in range(len(alive))], self.budget):
            chosen = [alive[i] for i in idx]
            x, kvs, lens_list = batch_state([(i, state[i]["x"], state[i]["kvs"]) for i in chosen])
            doc_lens = torch.tensor(lens_list, device=eng.device)
            sx = suffix_through(eng, cache, P, kvs, doc_lens, self.suffix_ids, 0, self.stages[1][0], None)
            width = max(n + S for n in lens_list)
            xj = x.new_zeros((len(chosen), width, x.shape[-1]))
            for local, ntok in enumerate(lens_list):
                xj[local, :ntok] = x[local, :ntok]
                xj[local, ntok : ntok + S] = sx[local]
            hkv, hd = kvs[0][0].shape[1], kvs[0][0].shape[-1]
            dummy = [(xj.new_zeros(len(chosen), hkv, width, hd), xj.new_zeros(len(chosen), hkv, width, hd))]
            xj, kvs2, lens, _valid = advance(eng, cache, P, [conts[i] for i in chosen], xj, dummy, self.stages[1][0] + 1, layer)
            sc = self.ridge_of(xj, lens)
            for local, doc_i in enumerate(chosen):
                ntok = int(lens[local])
                joint[doc_i] = {
                    "x": xj[local, :ntok].contiguous(),
                    "kvs": [(kk[local, :, :ntok].contiguous(), vv[local, :, :ntok].contiguous()) for kk, vv in kvs2],
                }
                scores[doc_i] = float(sc[local])
        t = _add(laps, "suffix_through_ridge", t)
        last = keep_top(scores, alive, self.ridge_keep)
        last_set = set(last)
        for i in alive:
            if i not in last_set:
                tier_key[i] = 3
                tier_score[i] = scores[i]
        exact = torch.full((len(conts),), float("-inf"))
        for idx in chunks([joint[last[i]]["x"].shape[0] for i in range(len(last))], self.budget):
            chosen = [last[i] for i in idx]
            x, kvs, _lens_list = batch_state([(i, joint[i]["x"], joint[i]["kvs"]) for i in chosen])
            x, kvs, lens, _valid = advance(eng, cache, P, [conts[i] for i in chosen], x, kvs, layer + 1, eng.L - 1)
            got = self.margins_of(x, lens)
            for local, doc_i in enumerate(chosen):
                exact[doc_i] = float(got[local])
                tier_key[doc_i] = 4
                tier_score[doc_i] = float(got[local])
        _add(laps, "layers_22_35", t)
        order = sorted(range(len(conts)), key=lambda i: (float(tier_key[i]), float(tier_score[i])), reverse=True)
        return order, exact, tier_key, tier_score


def slice_rows(x, kvs, lens, idx):
    from paper_eval import slice_state

    return slice_state(x, kvs, lens, idx)


def probe_forward(eng, probe, cache, P, kvs, lens, valid, end_layer: int, x=None, start_layer: int = 0):
    """Probe residual from `start_layer` through `end_layer`. Pass `x` to continue."""
    B, S = lens.shape[0], probe.emb.shape[0]
    hq, hd = eng.hq, eng.hd
    if x is None:
        if start_layer != 0:
            raise RuntimeError("continuing a probe requires its residual")
        x = probe.emb[None].expand(B, S, -1).clone()
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
            if j < start_layer:
                continue
            if j > end_layer:
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
            a = F.scaled_dot_product_attention(
                q, kf, vf, attn_mask=mask[:, None], enable_gqa=True, scale=attn.scaling
            )
            x = x + ad.o(a.transpose(1, 2).reshape(B, S, -1))
            h2 = layer_m.post_attention_layernorm(x)
            x = x + ad.down(F.silu(ad.gate(h2)) * ad.up(h2))
            if j == end_layer:
                return probe.heads[str(j)](x[:, -1].float()), x
    raise RuntimeError(f"probe has no layer {end_layer}")


class ContCascade(Cascade):
    """Same cuts as Cascade, but the layer-15 probe continues the layer-9 residual."""

    def _probe_stage(self, cache, P, docs, state, alive, depth, layer, k, stage_score):
        eng = self.eng
        scores = stage_score.clone()
        saved = getattr(self, "_probe_x", {})
        for idx in chunks([state[alive[i]]["x"].shape[0] for i in range(len(alive))], self.budget):
            chosen = [alive[i] for i in idx]
            x, kvs, lens_list = batch_state([(i, state[i]["x"], state[i]["kvs"]) for i in chosen])
            x, kvs, lens, valid = advance(eng, cache, P, [docs[i] for i in chosen], x, kvs, depth + 1, layer)
            px = torch.stack([saved[i] for i in chosen]) if saved else None
            sc, new_x = probe_forward(
                eng, self.probe, cache, P, kvs, lens, valid, layer, px,
                0 if px is None else depth + 1,
            )
            for local, doc_i in enumerate(chosen):
                ntok = int(lens[local])
                state[doc_i]["x"] = x[local, :ntok].contiguous()
                state[doc_i]["kvs"] = [(kk[local, :, :ntok].contiguous(), vv[local, :, :ntok].contiguous()) for kk, vv in kvs]
                saved[doc_i] = new_x[local].detach()
                scores[doc_i] = float(sc[local])
        self._probe_x = saved
        return keep_top(scores, alive, k), scores

    def run(self, row: dict):
        self._probe_x = {}
        eng = self.eng
        S = len(self.suffix_ids)
        docs = [c[:-S] for c in row["conts"]]
        n = len(docs)
        cache, P = eng.prefix(row["prefix"], Plan())
        state = {}
        stage_score = torch.zeros(n)
        for idx in chunks([max(1, len(d)) for d in docs], self.budget):
            x, kvs, lens, valid = advance(eng, cache, P, [docs[i] for i in idx], None, None, 0, self.stages[0][0])
            scores, px = probe_forward(eng, self.probe, cache, P, kvs, lens, valid, self.stages[0][0], None, 0)
            local = {doc_i: t for t, doc_i in enumerate(idx)}
            for doc_i, xi, kvi in slice_rows(x, kvs, lens, idx):
                state[doc_i] = {"x": xi, "kvs": kvi}
                self._probe_x[doc_i] = px[local[doc_i]].detach()
                stage_score[doc_i] = float(scores[local[doc_i]])
        tier_key = torch.zeros(n)
        tier_score = stage_score.clone()
        alive = keep_top(stage_score, list(range(n)), self.stages[0][1])
        for i in range(n):
            if i not in set(alive):
                tier_key[i] = 1
                self._probe_x.pop(i, None)
        depth = self.stages[0][0]
        for t, (layer, k) in enumerate(self.stages[1:], start=2):
            alive, stage_score = self._probe_stage(cache, P, docs, state, alive, depth, layer, k, stage_score)
            for i in range(n):
                if tier_key[i] == 0 and i not in set(alive):
                    tier_key[i] = t
                    tier_score[i] = stage_score[i]
                    self._probe_x.pop(i, None)
            depth = layer
        del state
        return self._finish(cache, P, row["conts"], alive, tier_key, tier_score)


def _rank_report(rows, orders, exacts) -> dict:
    ref = torch.load(DATA / "runs/check/collect_4b_b8000.pt", map_location="cpu")
    jac, cover, gaps = [], [], []
    qrels, got, teacher = {}, {}, {}
    for row, order, exact in zip(rows, orders, exacts):
        margins = ref["queries"][row["qid"]]["margins"]
        ref_order = sorted(range(100), key=lambda i: float(margins[i]), reverse=True)
        jac.append(jaccard(order, ref_order))
        kept = {i for i in range(100) if float(exact[i]) > float("-inf")}
        cover.append(float(set(ref_order[:10]) <= kept))
        for i in kept:
            gaps.append(abs(float(exact[i]) - float(margins[i])))
        got[row["qid"]] = [row["ids"][i] for i in order]
        teacher[row["qid"]] = [row["ids"][i] for i in ref_order]
        if row.get("qrels"):
            qrels[row["qid"]] = row["qrels"]
    report = {
        "n": len(rows),
        "jaccard@10": sum(jac) / len(jac),
        "cover": sum(cover) / len(cover),
        "cascade_ndcg@10": ndcg(qrels, got),
        "teacher_ndcg@10": ndcg(qrels, teacher),
        "finalist_margin_max_abs": max(gaps) if gaps else None,
        "finalist_margin_mean_abs": sum(gaps) / len(gaps) if gaps else None,
        "n_finalist_margins": len(gaps),
    }
    report["drop"] = report["teacher_ndcg@10"] - report["cascade_ndcg@10"]
    return report


@torch.no_grad()
def catchup_mode(limit: int, time_n: int, score: bool = True) -> None:
    eng, cas, _suffix = load_all()
    catch = Catchup(eng, cas.probe, cas.beta, cas.suffix_ids, cas.budget)
    if time_n:
        rows = torch.load(DATA / "pilot.pt")[: time_n + 2]
        for row in rows[:2]:
            catch.run(row)
            score_query(eng, row, Plan(), 4000)
        base, ours = [], []
        for row in rows[2:]:
            base.append(timed(lambda row=row: score_query(eng, row, Plan(), 4000)))
            ours.append(timed(lambda row=row: catch.run(row)))
            print(f"time {row['qid']} base {base[-1]:.0f} catch {ours[-1]:.0f}", flush=True)
        time_report = {
            "n": len(base),
            "slice": "pilot[2:n+2], warmup 2",
            "baseline_mean_ms": sum(base) / len(base),
            "catchup_mean_ms": sum(ours) / len(ours),
        }
        time_report["mean_speedup"] = time_report["baseline_mean_ms"] / time_report["catchup_mean_ms"]
        time_report["probe"] = "layer-15 head continues the layer-9 residual"
        dest = DATA / "paper_catchup_time.json"
        dest.write_text(json.dumps(time_report, indent=2) + "\n")
        print(json.dumps(time_report, indent=2), flush=True)
    if not score:
        return
    rows = torch.load(DATA / "check.pt")
    if limit:
        rows = rows[:limit]
    orders, exacts = [], []
    for n, row in enumerate(rows, 1):
        order, exact, _k, _s = catch.run(row)
        orders.append(order)
        exacts.append(exact)
        if n % 20 == 0 or n == len(rows):
            print(f"catchup {n}/{len(rows)}", flush=True)
    report = _rank_report(rows, orders, exacts)
    (DATA / "paper_catchup.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


@torch.no_grad()
def numeric_mode() -> None:
    row = torch.load(DATA / "check.pt")[0]
    S = row["suffix_len"]
    suffix_ids = row["conts"][0][-S:]
    lengths = [len(c) - S for c in row["conts"]]
    order = sorted(range(100), key=lambda i: lengths[i])
    pick = [order[0], order[len(order) // 2], order[-1]]
    print(f"qid {row['qid']} docs {pick} lens {[lengths[i] for i in pick]}", flush=True)
    out = {"qid": row["qid"], "docs": pick, "doc_lens": [lengths[i] for i in pick], "runs": {}}
    for name, bf16 in (("bf16", True), ("fp32", False)):
        if bf16:
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        eng = load_engine(MODEL, bf16=bf16)
        cache, P = eng.prefix(row["prefix"], Plan())
        docs = [row["conts"][i][:-S] for i in pick]
        conts = [row["conts"][i] for i in pick]
        ends = [0, 15, eng.L - 1]
        layers = {}
        for end in ends:
            xd, kvs, lens, _v = advance(eng, cache, P, docs, None, None, 0, end)
            xj, _kj, lensj, _vj = advance(eng, cache, P, conts, None, None, 0, end)
            doc_gap = []
            for local, ntok in enumerate(lens.tolist()):
                doc_gap.append(float((xd[local, :ntok] - xj[local, :ntok]).abs().max()))
            sx = suffix_through(eng, cache, P, kvs, lens, suffix_ids, 0, end, None)
            suf_gaps = []
            for local, ntok in enumerate(lens.tolist()):
                suf_gaps.append(float((sx[local] - xj[local, ntok : ntok + S]).abs().max()))
            rec = {"doc_hidden_max_abs": max(doc_gap), "suffix_hidden_max_abs": max(suf_gaps)}
            if end == eng.L - 1:
                mj = _margin(eng, xj[torch.arange(len(pick), device=eng.device), lensj - 1])
                ms = _margin(eng, sx[:, -1])
                rec["margin_max_abs"] = float((mj - ms).abs().max())
                rec["margin_per_doc"] = [float(v) for v in (mj - ms).abs()]
            layers[str(end)] = rec
            print(f"{name} through layer {end} doc {rec['doc_hidden_max_abs']:.6f} suffix {rec['suffix_hidden_max_abs']:.6f}"
                  + (f" margin {rec['margin_max_abs']:.6f}" if "margin_max_abs" in rec else ""), flush=True)
        out["runs"][name] = {"tf32": bool(torch.backends.cuda.matmul.allow_tf32), "layers": layers}
        del eng
        torch.cuda.empty_cache()
    (DATA / "paper_numeric.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


@torch.no_grad()
def probecont_mode(n: int) -> None:
    eng, cas, _s = load_all()
    row = torch.load(DATA / "pilot.pt")[0]
    S = len(cas.suffix_ids)
    docs = [c[:-S] for c in row["conts"]]
    cache, P = eng.prefix(row["prefix"], Plan())
    worst = 0.0
    for idx in chunks([max(1, len(d)) for d in docs], 4000):
        x, kvs, lens, valid = advance(eng, cache, P, [docs[i] for i in idx], None, None, 0, 15)
        full = probe_at(eng, cas.probe, cache, P, kvs, lens, valid, 15)
        _mid, px = probe_forward(eng, cas.probe, cache, P, kvs, lens, valid, 9, None, 0)
        cont, _px = probe_forward(eng, cas.probe, cache, P, kvs, lens, valid, 15, px, 10)
        worst = max(worst, float((full - cont).abs().max()))
    print(f"probe continue vs fresh layer15 max abs {worst:.6f}", flush=True)
    if worst > 0.05:
        raise SystemExit("continued probe does not match the fresh probe; not timing it")
    cont_cas = ContCascade(eng, cas.probe, cas.beta, cas.suffix_ids, cas.budget)
    rows = torch.load(DATA / "pilot.pt")[: n + 2]
    for row in rows[:2]:
        cas.run(row)
        cont_cas.run(row)
    fresh, kept = [], []
    for row in rows[2:]:
        fresh.append(timed(lambda row=row: cas.run(row)))
        kept.append(timed(lambda row=row: cont_cas.run(row)))
        print(f"probe {row['qid']} fresh {fresh[-1]:.0f} continued {kept[-1]:.0f}", flush=True)
    report = {
        "agree_max_abs_layer15": worst,
        "n": len(fresh),
        "fresh_probe_mean_ms": sum(fresh) / len(fresh),
        "continued_probe_mean_ms": sum(kept) / len(kept),
    }
    report["mean_speedup_vs_fresh_cascade"] = report["fresh_probe_mean_ms"] / report["continued_probe_mean_ms"]
    (DATA / "paper_probecont.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def ridge_mode() -> None:
    import spec5
    from spec import fit_ridge, rms

    spec5.set_model("4b")
    frun, fhid = spec5.load_collect("fit")
    fq = list(fhid["hidden"])
    yf = torch.tensor([frun["queries"][q]["margins"] for q in fq], dtype=torch.float32)
    lay = fhid["layers"]
    cut = int(0.85 * len(fq))
    held = {}
    betas = {}
    for layer in (9, 15, 21):
        k = lay.index(layer)
        X = torch.stack([rms(fhid["hidden"][q][:, k].float()) for q in fq])
        best = None
        for lam in (1e-4, 1e-3, 1e-2):
            beta = fit_ridge(X[:cut], yf[:cut], lam)
            pred = X[cut:] @ beta
            c = float(np.mean([np.corrcoef(pred[i].numpy(), yf[cut + i].numpy())[0, 1] for i in range(len(pred))]))
            if best is None or c > best[0]:
                best = (c, lam, beta)
        betas[layer] = fit_ridge(X, yf, best[1])
        held[str(layer)] = {"lam": best[1], "heldout_within_query_corr": best[0]}
        print(f"ridge@{layer} lam {best[1]} held-out corr {best[0]:.4f}", flush=True)
    del fhid, frun
    sp = spec5.Split("check")
    scores = []
    for layer in (9, 15, 21):
        k = sp.full["layers"].index(layer)
        X = torch.stack([rms(sp.full["hidden"][q][:, k].float()) for q in sp.qids])
        scores.append((X @ betas[layer]).numpy())
    keeps = (60, 40, 15)
    mac = spec5.run(sp, [
        ("full", 0, 9, scores[0], 60, "ridge"),
        ("full", 0, 15, scores[1], 40, "ridge"),
        ("full", 0, 21, scores[2], 15, "ridge"),
    ])
    Q = len(sp.qids)
    alive = np.ones((Q, 100), dtype=bool)
    key = np.zeros((Q, 100))
    tier = np.zeros((Q, 100))
    for t, (sc, K) in enumerate(zip(scores, keeps)):
        s = np.where(alive, sc, -np.inf)
        kth = -np.sort(-s, 1)[:, K - 1 : K]
        keep = alive & (s >= kth)
        dropped = alive & ~keep
        key = np.where(dropped, t + 1, key)
        tier = np.where(dropped, sc, tier)
        alive = keep
    key = np.where(alive, 4, key)
    tier = np.where(alive, sp.margin, tier)
    order = np.lexsort((-tier, -key), axis=1)
    qrels, got, teacher = {}, {}, {}
    for qi, q in enumerate(sp.qids):
        r = sp.rows[q]
        got[q] = [r["ids"][int(i)] for i in order[qi]]
        teacher[q] = [r["ids"][int(i)] for i in sp.ref_rank[qi]]
        if r.get("qrels"):
            qrels[q] = r["qrels"]
    report = {
        "rule": "ridge@9/60 -> ridge@15/40 -> ridge@21/15, suffix attached from layer 0",
        "fit_queries": 600,
        "heldout": held,
        "mac_frac": mac["mac_frac"],
        "spec5_ndcg@10": mac["ndcg@10"],
        "spec5_drop": mac["drop"],
        "jaccard@10": mac["jaccard@10"],
        "cover": mac["cover"],
        "mteb_cascade_ndcg@10": ndcg(qrels, got),
        "mteb_teacher_ndcg@10": ndcg(qrels, teacher),
    }
    report["mteb_drop"] = report["mteb_teacher_ndcg@10"] - report["mteb_cascade_ndcg@10"]
    (DATA / "paper_ridge.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def _per_query_ndcg(qrels: dict, rankings: dict) -> dict[str, float]:
    import pytrec_eval

    results = {}
    for qid, doc_ids in rankings.items():
        n = len(doc_ids)
        results[qid] = {doc_id: float(n - rank) for rank, doc_id in enumerate(doc_ids)}
    clean = {qid: docs for qid, docs in qrels.items() if qid in results and docs}
    results = {qid: docs for qid, docs in results.items() if qid in clean}
    evaluator = pytrec_eval.RelevanceEvaluator(clean, {"ndcg_cut.10"})
    scores = evaluator.evaluate(results)
    return {qid: float(row["ndcg_cut_10"]) for qid, row in scores.items()}


def _bootstrap(delta: np.ndarray, draws: int = 10000) -> dict:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(delta), size=(draws, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    # Two-sided: share of bootstrap means whose sign disagrees with the observed mean.
    observed = float(delta.mean())
    if observed == 0:
        p = 1.0
    else:
        p = float(np.mean(means * observed <= 0) * 2)
    return {
        "n": int(len(delta)),
        "mean_cascade_minus_teacher": observed,
        "ci95": [float(lo), float(hi)],
        "bootstrap_p_twosided": min(1.0, p),
        "draws": draws,
    }


def boot_mode() -> None:
    import mteb

    manifest = load_manifest()
    qids = [str(q) for q in manifest["arabic_dev_ids"]]
    saved = {}
    for line in (DATA / "paper_dev.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            saved[row["qid"]] = row["order"]
    teacher_rows = {}
    for line in (OUT / "dev_4b.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            teacher_rows[row["qid"]] = row
    task = mteb.get_tasks(tasks=["MIRACLRetrieval"], languages=["ara"])[0]
    task.load_data()
    raw = task.dataset["ar"]["dev"]["relevant_docs"]
    qrels = {str(q): {str(d): int(r) for d, r in docs.items()} for q, docs in raw.items() if str(q) in set(qids) and docs}
    cascade = {q: saved[q] for q in qids if q in saved}
    teacher = {}
    for q in cascade:
        scores = teacher_rows[q]["scores"]
        teacher[q] = sorted(scores, key=lambda d: (float(scores[d]), d), reverse=True)
    c_nd = _per_query_ndcg(qrels, cascade)
    t_nd = _per_query_ndcg(qrels, teacher)
    keys = [q for q in c_nd if q in t_nd]
    delta = np.array([c_nd[q] - t_nd[q] for q in keys])
    dev = _bootstrap(delta)
    dev["mean_cascade"] = float(np.mean([c_nd[q] for q in keys]))
    dev["mean_teacher"] = float(np.mean([t_nd[q] for q in keys]))
    report = {"dev": dev}
    dl19_path = DATA / "paper_dl19_queries.jsonl"
    if dl19_path.exists():
        orders = {json.loads(line)["qid"]: json.loads(line)["order"] for line in dl19_path.read_text().splitlines() if line.strip()}
        teacher_e = {}
        for line in (OUT / "english/TRECDL2019_4b.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                teacher_e[row["qid"]] = row
        task = mteb.get_tasks(tasks=["TRECDL2019"])[0]
        task.load_data()
        split = task.dataset["default"]["test"] if "default" in task.dataset else task.dataset["test"]
        raw = split["relevant_docs"]
        qrels_e = {str(q): {str(d): int(r) for d, r in docs.items()} for q, docs in raw.items()}
        cascade_e = orders
        teacher_rank = {}
        for q in orders:
            scores = teacher_e[q]["scores"]
            teacher_rank[q] = sorted(scores, key=lambda d: (float(scores[d]), d), reverse=True)
        c_nd = _per_query_ndcg(qrels_e, cascade_e)
        t_nd = _per_query_ndcg(qrels_e, teacher_rank)
        keys = [q for q in c_nd if q in t_nd]
        delta = np.array([c_nd[q] - t_nd[q] for q in keys])
        dl = _bootstrap(delta)
        dl["mean_cascade"] = float(np.mean([c_nd[q] for q in keys]))
        dl["mean_teacher"] = float(np.mean([t_nd[q] for q in keys]))
        report["dl19"] = dl
    (DATA / "paper_bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


@torch.no_grad()
def dl19_orders_mode() -> None:
    from paper_eval import build_rows

    eng, cas, _s = load_all()
    manifest = load_manifest()
    raw_ids = manifest["english_pool"]["TRECDL2019"]
    qids = [str(q["id"] if isinstance(q, dict) else q) for q in raw_ids]
    rows = build_rows(OUT / "english/TRECDL2019_top100.jsonl", qids)
    dest = DATA / "paper_dl19_queries.jsonl"
    with dest.open("w") as fh:
        for n, row in enumerate(rows, 1):
            order, _exact, _k, _s = cas.run(row)
            fh.write(json.dumps({"qid": row["qid"], "order": [row["ids"][i] for i in order]}) + "\n")
            if n % 10 == 0 or n == len(rows):
                print(f"dl19 orders {n}/{len(rows)}", flush=True)
    print(f"wrote {dest}", flush=True)


def main() -> None:
    mode = sys.argv[1]
    if mode == "catchup":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 400
        time_n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        catchup_mode(limit, time_n)
    elif mode == "catchup_time":
        catchup_mode(0, int(sys.argv[2]) if len(sys.argv) > 2 else 30, score=False)
    elif mode == "numeric":
        numeric_mode()
    elif mode == "probecont":
        probecont_mode(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif mode == "ridge":
        ridge_mode()
    elif mode == "boot":
        boot_mode()
    elif mode == "dl19orders":
        dl19_orders_mode()
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
