#!/usr/bin/env python3
"""Shrink TailCatch after layer 15 without padding and without cutting finalist tails.

Lever 1 runs doc[192:] and the suffix as one sequence, and runs the 9 suffix
tokens only on the short survivors. Layers 16-35 stay on the original 4000
budget. Lever 2, after a chosen layer, advances only the suffix (or the last
16 document tokens plus the suffix) and keeps their original positions.

`collect` records the TailCatch stage scores and sweeps tighter keep sizes on
CPU. Writes week3_fast_k.json and week3_fast_tail.json. Does not overwrite
older results and does not open the holdout, dev, or DL19.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import core
from core import Plan
from fit_rules import ndcg
from next_exps import ContCascade, probe_forward, slice_rows
from paper_eval import DATA, advance, batch_state, chunks, jaccard, keep_top, load_all, suffix_through, timed
from week3_tail import CAP, TailCatch
from week_exps import load_saved_probe, tail_through

K9 = (45, 48, 50, 60)
K15 = (24, 25, 28, 32, 40)
K21 = (10, 11, 12, 15)


def probe_heads(model, row, cap: int = CAP):
    """Head probes. Same cuts TailCatch uses, and the score vectors before they are overwritten."""
    eng = model.eng
    docs = [c[: -len(model.suffix_ids)] for c in row["conts"]]
    heads = [d[:cap] for d in docs]
    n = len(docs)
    cache, P = eng.prefix(row["prefix"], Plan())
    model._probe_x = {}
    state = {}
    stage_score = torch.zeros(n)
    for idx in chunks([max(1, len(d)) for d in heads], model.budget):
        x, kvs, lens, valid = advance(eng, cache, P, [heads[i] for i in idx], None, None, 0, model.stages[0][0])
        scores, px = probe_forward(eng, model.probe, cache, P, kvs, lens, valid, model.stages[0][0], None, 0)
        local = {doc_i: t for t, doc_i in enumerate(idx)}
        for doc_i, xi, kvi in slice_rows(x, kvs, lens, idx):
            state[doc_i] = {"x": xi, "kvs": kvi}
            model._probe_x[doc_i] = px[local[doc_i]].detach()
            stage_score[doc_i] = float(scores[local[doc_i]])
    s9 = stage_score.clone()
    tier_key = torch.zeros(n)
    tier_score = stage_score.clone()
    alive = keep_top(stage_score, list(range(n)), model.stages[0][1])
    for i in range(n):
        if i not in set(alive):
            tier_key[i] = 1
            model._probe_x.pop(i, None)
            state.pop(i, None)
    depth = model.stages[0][0]
    for tier_i, (layer, k) in enumerate(model.stages[1:], start=2):
        alive, stage_score = ContCascade._probe_stage(
            model, cache, P, heads, state, alive, depth, layer, k, stage_score
        )
        for i in list(state):
            if i not in set(alive):
                if tier_key[i] == 0:
                    tier_key[i] = tier_i
                    tier_score[i] = stage_score[i]
                model._probe_x.pop(i, None)
                state.pop(i, None)
        depth = layer
    return cache, P, docs, alive, tier_key, tier_score, state, s9, stage_score.clone(), depth


class RecordingTail(TailCatch):
    """TailCatch, plus the three stage scores a tighter keep tuple needs."""

    def run(self, row):
        cache, P, docs, alive, tier_key, tier_score, state, s9, s15, depth = probe_heads(self, row)
        self.s9 = s9
        self.s15 = s15
        self._long = sum(1 for i in alive if len(docs[i]) > CAP)
        state = self._catch_tails(cache, P, docs, state, alive, depth)
        return self._finish_record(cache, P, row["conts"], alive, tier_key, tier_score, state)

    def _finish_record(self, cache, P, conts, alive, tier_key, tier_score, state):
        eng = self.eng
        S = len(self.suffix_ids)
        layer = self.ridge_layer
        scores = tier_score.clone()
        joint = {}
        if not alive:
            self.s21 = scores.clone()
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
            xj, _kvs2, lens, _valid = advance(eng, cache, P, [conts[i] for i in chosen], xj, dummy, self.stages[1][0] + 1, layer)
            sc = self.ridge_of(xj, lens)
            for local, doc_i in enumerate(chosen):
                ntok = int(lens[local])
                joint[doc_i] = {"x": xj[local, :ntok].contiguous(), "tokens": conts[doc_i]}
                scores[doc_i] = float(sc[local])
        self.s21 = scores.clone()
        last = keep_top(scores, alive, self.ridge_keep)
        last_set = set(last)
        for i in alive:
            if i not in last_set:
                tier_key[i] = 3
                tier_score[i] = scores[i]
        exact = torch.full((len(conts),), float("-inf"))
        for idx in chunks([joint[i]["x"].shape[0] for i in last], self.budget):
            chosen = [last[i] for i in idx]
            states = [(i, joint[i]["x"], joint[i]["tokens"]) for i in chosen]
            x, _kvs, lens, _valid = _advance_states(eng, cache, P, states, layer + 1, eng.L - 1)
            got = self.margins_of(x, lens)
            for local, doc_i in enumerate(chosen):
                exact[doc_i] = float(got[local])
                tier_key[doc_i] = 4
                tier_score[doc_i] = float(got[local])
        order = sorted(range(len(conts)), key=lambda i: (float(tier_key[i]), float(tier_score[i])), reverse=True)
        return order, exact, tier_key, tier_score


def _advance_states(eng, cache, P, states, start, end):
    """states: (doc_i, x, tokens). One already-packed group."""
    tokens = [s[2] for s in states]
    W = max(len(t) for t in tokens)
    x0 = states[0][1]
    x = x0.new_zeros((len(states), W, x0.shape[-1]))
    for i, (_doc, xi, toks) in enumerate(states):
        x[i, : len(toks)] = xi[: len(toks)]
    hkv = cache[0][0].shape[1]
    hd = cache[0][0].shape[-1]
    dummy = [(x.new_zeros(len(states), hkv, W, hd), x.new_zeros(len(states), hkv, W, hd))]
    return advance(eng, cache, P, tokens, x, dummy, start, end)


class MergedTail(TailCatch):
    """Lever 1, and optionally a suffix-only tail after `freeze`."""

    def __init__(self, *args, freeze: int | None = None, active: int = 9, keeps: tuple[int, int, int] | None = None):
        super().__init__(*args)
        self.freeze = freeze
        self.active = active
        self.keeps = keeps
        if keeps is not None:
            k9, k15, k21 = keeps
            self.stages = [(9, k9), (15, k15)]
            self.ridge_keep = k21
        self._long = 0
        self.late_chunks = 0
        self.final_chunks = 0
        self.capture_h = None
        self.final_budget = None
        self.head_cap = CAP
        self.early_compact = False

    def run(self, row):
        eng = self.eng
        S = len(self.suffix_ids)
        if self.early_compact:
            import paper_eval as pe
            pe.COMPACT = "all"
        try:
            cache, P, docs, alive, tier_key, tier_score, state, _s9, _s15, depth = probe_heads(self, row, self.head_cap)
        finally:
            if self.early_compact:
                import paper_eval as pe
                pe.COMPACT = None
        longs = [i for i in alive if len(docs[i]) > self.head_cap]
        shorts = [i for i in alive if i not in set(longs)]
        self._long = len(longs)
        full = {}
        full.update(self._long_once(cache, P, docs, state, longs, depth))
        full.update(self._short_suffix(cache, P, docs, state, shorts))
        return self._late(cache, P, row["conts"], alive, tier_key, tier_score, full)

    def _long_once(self, cache, P, docs, state, longs, end):
        if not longs:
            return {}
        eng = self.eng
        S = len(self.suffix_ids)
        out = {}
        ids = list(longs)
        for loc in chunks([len(docs[i]) for i in ids], self.budget):
            group = [ids[i] for i in loc]
            seqs = [docs[i][state[i]["x"].shape[0] :] + self.suffix_ids for i in group]
            px, pkvs, plens = batch_state([(i, state[i]["x"], state[i]["kvs"]) for i in group])
            pre = torch.tensor(plens, device=eng.device)
            tx, _tkvs, _tlens = tail_through(eng, cache, P, pkvs, pre, seqs, end)
            for local, doc_i in enumerate(group):
                n = int(pre[local])
                m = len(seqs[local])
                x = torch.cat([px[local, :n], tx[local, :m]], 0).contiguous()
                out[doc_i] = x
        return out

    def _short_suffix(self, cache, P, docs, state, shorts):
        if not shorts:
            return {}
        eng = self.eng
        S = len(self.suffix_ids)
        px, pkvs, plens = batch_state([(i, state[i]["x"], state[i]["kvs"]) for i in shorts])
        pre = torch.tensor(plens, device=eng.device)
        sx = suffix_through(eng, cache, P, pkvs, pre, self.suffix_ids, 0, self.stages[1][0], None)
        out = {}
        for local, doc_i in enumerate(shorts):
            n = int(pre[local])
            out[doc_i] = torch.cat([px[local, :n], sx[local]], 0).contiguous()
        return out

    def _late(self, cache, P, conts, alive, tier_key, tier_score, full):
        eng = self.eng
        S = len(self.suffix_ids)
        for i in alive:
            if full[i].shape[0] != len(conts[i]):
                raise RuntimeError(f"doc {i} hidden {full[i].shape[0]} != tokens {len(conts[i])}")
        scores = tier_score.clone()
        held = {}
        nchunks = 0
        ids = list(alive)
        lengths = [full[i].shape[0] for i in ids]
        for loc in chunks(lengths, self.budget):
            group = [ids[i] for i in loc]
            nchunks += 1
            x, _kvs, lens, _valid = _advance_states(
                eng, cache, P, [(i, full[i], conts[i]) for i in group], self.stages[1][0] + 1, self.ridge_layer
            )
            sc = self.ridge_of(x, lens)
            for local, doc_i in enumerate(group):
                held[doc_i] = x[local, : int(lens[local])].contiguous()
                scores[doc_i] = float(sc[local])
        self.late_chunks = nchunks
        last = keep_top(scores, alive, self.ridge_keep)
        last_set = set(last)
        for i in alive:
            if i not in last_set:
                tier_key[i] = 3
                tier_score[i] = scores[i]
        exact = torch.full((len(tier_key),), float("-inf"))
        if last:
            final_ids = list(last)
            final_lengths = [held[i].shape[0] for i in final_ids]
            late_budget = self.budget if self.final_budget is None else self.final_budget
            if self.freeze is None:
                nchunks = 0
                for loc in chunks(final_lengths, late_budget):
                    group = [final_ids[i] for i in loc]
                    nchunks += 1
                    x, _kvs, lens, _valid = _advance_states(
                        eng, cache, P, [(i, held[i], conts[i]) for i in group], self.ridge_layer + 1, eng.L - 1
                    )
                    got = self.margins_of(x, lens)
                    for local, doc_i in enumerate(group):
                        exact[doc_i] = float(got[local])
                self.final_chunks = nchunks
            else:
                mid = {}
                nchunks = 0
                for loc in chunks(final_lengths, late_budget):
                    group = [final_ids[i] for i in loc]
                    nchunks += 1
                    x, _kvs, lens, _valid = _advance_states(
                        eng, cache, P, [(i, held[i], conts[i]) for i in group], self.ridge_layer + 1, self.freeze
                    )
                    for local, doc_i in enumerate(group):
                        mid[doc_i] = x[local, : int(lens[local])].contiguous()
                        if self.capture_h is not None:
                            self.capture_h[doc_i] = mid[doc_i][-1].detach().float().cpu()
                rows = []
                order_docs = []
                for doc_i in final_ids:
                    x = mid[doc_i]
                    n_doc = x.shape[0] - S
                    take = S if self.active <= S else min(self.active - S, n_doc) + S
                    pos0 = x.shape[0] - take
                    rows.append((x[pos0:].contiguous(), pos0))
                    order_docs.append(doc_i)
                got = _active_margin(eng, cache, P, rows, self.freeze + 1, eng.L - 1)
                self.final_chunks = nchunks + 1
                for doc_i, margin in zip(order_docs, got):
                    exact[doc_i] = float(margin)
            for doc_i in last:
                tier_key[doc_i] = 4
                tier_score[doc_i] = exact[doc_i]
        order = sorted(range(len(tier_key)), key=lambda i: (float(tier_key[i]), float(tier_score[i])), reverse=True)
        return order, exact, tier_key, tier_score


def _active_margin(eng, cache, P, rows, start, end):
    """rows: (x, position of x[0]). Advance only these tokens. They see the prefix and themselves."""
    W = max(xi.shape[0] for xi, _pos in rows)
    B = len(rows)
    x0 = rows[0][0]
    x = x0.new_zeros((B, W, x0.shape[-1]))
    pos = torch.zeros((B, W), dtype=torch.long, device=x.device)
    lens = []
    for i, (xi, pos0) in enumerate(rows):
        n = xi.shape[0]
        x[i, :n] = xi
        pos[i, :n] = P + pos0 + torch.arange(n, device=x.device)
        lens.append(n)
    lens_t = torch.tensor(lens, device=x.device)
    valid = torch.arange(W, device=x.device)[None] < lens_t[:, None]
    causal = torch.ones((W, W), dtype=torch.bool, device=x.device).tril()[None] & valid[:, None, :]
    mask = torch.cat([torch.ones((B, W, P), dtype=torch.bool, device=x.device), causal], 2)
    cos, sin = eng.base.rotary_emb(x, pos)
    plan = Plan()
    for j in range(start, end + 1):
        layer = eng.base.layers[j]
        q, k, v = eng._proj(layer, layer.input_layernorm(x), plan)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        pk, pv = cache[j]
        kf = torch.cat([pk.expand(B, -1, -1, -1), k], 2)
        vf = torch.cat([pv.expand(B, -1, -1, -1), v], 2)
        a = F.scaled_dot_product_attention(
            q, kf, vf, attn_mask=mask[:, None], enable_gqa=True, scale=layer.self_attn.scaling
        )
        x = x + layer.self_attn.o_proj(a.transpose(1, 2).reshape(B, W, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = x * valid[..., None]
    last = eng.base.norm(x[torch.arange(B, device=x.device), lens_t - 1]).float()
    return last @ eng.w


def _overlap(a, b) -> float:
    return len(set(a[:10]) & set(b[:10])) / 10


def _gap(ea, eb):
    both = [i for i in range(len(ea)) if float(ea[i]) > float("-inf") and float(eb[i]) > float("-inf")]
    if not both:
        return None
    return max(abs(float(ea[i]) - float(eb[i])) for i in both)


def _replay(s9, s15, s21, exact, k9, k15, k21):
    """Tighter keeps. A new finalist that never received an exact margin stays on its ridge score."""
    n = s9.shape[0]
    tier_key = torch.zeros(n)
    tier_score = s9.clone()
    alive = keep_top(s9, list(range(n)), k9)
    alive_set = set(alive)
    for i in range(n):
        if i not in alive_set:
            tier_key[i] = 1
    alive = keep_top(s15, alive, k15)
    alive_set = set(alive)
    for i in range(n):
        if tier_key[i] == 0 and i not in alive_set:
            tier_key[i] = 2
            tier_score[i] = s15[i]
    last = keep_top(s21, alive, k21)
    last_set = set(last)
    proxy = 0
    for i in alive:
        if i not in last_set:
            tier_key[i] = 3
            tier_score[i] = s21[i]
    for i in last:
        if float(exact[i]) > float("-inf"):
            tier_key[i] = 4
            tier_score[i] = exact[i]
        else:
            tier_key[i] = 3
            tier_score[i] = s21[i]
            proxy += 1
    order = sorted(range(n), key=lambda i: (float(tier_key[i]), float(tier_score[i])), reverse=True)
    return order, proxy


def _rank_drop(rows, orders, ref):
    qrels, got, teacher = {}, {}, {}
    for row, order in zip(rows, orders):
        margins = ref["queries"][row["qid"]]["margins"]
        ref_order = sorted(range(100), key=lambda i: float(margins[i]), reverse=True)
        got[row["qid"]] = [row["ids"][i] for i in order]
        teacher[row["qid"]] = [row["ids"][i] for i in ref_order]
        if row.get("qrels"):
            qrels[row["qid"]] = row["qrels"]
    teacher_nd = ndcg(qrels, teacher)
    cascade_nd = ndcg(qrels, got)
    return {"cascade_ndcg@10": cascade_nd, "teacher_ndcg@10": teacher_nd, "drop": teacher_nd - cascade_nd}


def collect(eng, cas, suffix_ids) -> None:
    probe = load_saved_probe(eng, suffix_ids, "q4_p1")
    model = RecordingTail(eng, probe, cas.beta, suffix_ids, cas.budget)
    rows = torch.load(DATA / "check.pt", weights_only=False)
    ref = torch.load(DATA / "runs/check/collect_4b_b8000.pt", map_location="cpu", weights_only=False)
    for row in rows[:2]:
        model.run(row)
    saved = []
    live_orders = []
    for n, row in enumerate(rows, 1):
        order, exact, _tk, _ts = model.run(row)
        saved.append({
            "s9": model.s9.clone(),
            "s15": model.s15.clone(),
            "s21": model.s21.clone(),
            "exact": exact.clone(),
        })
        live_orders.append(order)
        if n % 25 == 0 or n == len(rows):
            print(f"collect {n}/{len(rows)}", flush=True)
    torch.save({"qids": [row["qid"] for row in rows], "scores": saved}, DATA / "week3_tail_stages.pt")
    live = _rank_drop(rows, live_orders, ref)
    grid = []
    for k9 in K9:
        for k15 in K15:
            for k21 in K21:
                orders = []
                proxy = 0
                for item in saved:
                    order, nproxy = _replay(item["s9"], item["s15"], item["s21"], item["exact"], k9, k15, k21)
                    orders.append(order)
                    proxy += nproxy
                ranked = _rank_drop(rows, orders, ref)
                grid.append({
                    "k": [k9, k15, k21],
                    "drop": ranked["drop"],
                    "cascade_ndcg@10": ranked["cascade_ndcg@10"],
                    "late_passage_layers": k15 * 6 + k21 * 14,
                    "new_finalists_without_exact": proxy,
                })
    ok = [g for g in grid if g["drop"] <= 0.0010]
    ok.sort(key=lambda g: (g["late_passage_layers"], g["drop"]))
    report = {
        "live_tail": live,
        "note": "A finalist created by a tighter layer-15 keep, who was not a finalist at (60, 40, 15), is ranked by the layer-21 ridge score. That proxy can only make the simulated drop worse.",
        "grid_n": len(grid),
        "within_0.001": ok,
        "best": ok[0] if ok else None,
    }
    (DATA / "week3_fast_k.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"live_drop": live["drop"], "best": report["best"], "within": len(ok)}, indent=2), flush=True)


def _screen_one(model, tail, rows, label):
    for row in rows[:2]:
        model.run(row)
    times, gaps, overs = [], [], []
    for row in rows[2:10]:
        times.append(timed(lambda row=row: model.run(row)))
        of, ef, _k, _s = model.run(row)
        ot, et, _k, _s = tail.run(row)
        gaps.append(_gap(ef, et))
        overs.append(_overlap(of, ot))
        print(
            f"{label} {row['qid']} {times[-1]:.0f} gap {gaps[-1]} overlap {overs[-1]:.2f} "
            f"long {model._long} late {model.late_chunks} final {model.final_chunks}",
            flush=True,
        )
    got = [g for g in gaps if g is not None]
    screen = {
        "label": label,
        "freeze": model.freeze,
        "active": model.active,
        "keeps": None if model.keeps is None else list(model.keeps),
        "ms": sum(times) / len(times),
        "margin_max_abs_vs_tail": max(got) if got else None,
        "top10_overlap_vs_tail": sum(overs) / len(overs),
    }
    print(json.dumps(screen), flush=True)
    return screen


def _check(model, rows, ref, label):
    for row in rows[:2]:
        model.run(row)
    times = [timed(lambda row=row: model.run(row)) for row in rows[2:10]]
    jac, cover = [], []
    qrels, got, teacher = {}, {}, {}
    for n, row in enumerate(rows, 1):
        order, exact, _k, _s = model.run(row)
        margins = ref["queries"][row["qid"]]["margins"]
        ref_order = sorted(range(100), key=lambda i: float(margins[i]), reverse=True)
        jac.append(jaccard(order, ref_order))
        kept = {i for i in range(100) if float(exact[i]) > float("-inf")}
        cover.append(float(set(ref_order[:10]) <= kept))
        got[row["qid"]] = [row["ids"][i] for i in order]
        teacher[row["qid"]] = [row["ids"][i] for i in ref_order]
        if row.get("qrels"):
            qrels[row["qid"]] = row["qrels"]
        if n % 25 == 0 or n == len(rows):
            print(f"{label} check {n}/{len(rows)} J {sum(jac) / len(jac):.3f} cover {sum(cover) / len(cover):.3f}", flush=True)
    teacher_nd = ndcg(qrels, teacher)
    cascade_nd = ndcg(qrels, got)
    return {
        "label": label,
        "check_n": len(rows),
        "check_mean_ms": sum(times) / len(times),
        "jaccard@10": sum(jac) / len(jac),
        "cover": sum(cover) / len(cover),
        "cascade_ndcg@10": cascade_nd,
        "teacher_ndcg@10": teacher_nd,
        "drop": teacher_nd - cascade_nd,
    }


def screen(eng, cas, suffix_ids) -> None:
    probe = load_saved_probe(eng, suffix_ids, "q4_p1")
    tail = TailCatch(eng, probe, cas.beta, suffix_ids, cas.budget)
    rows = torch.load(DATA / "check.pt", weights_only=False)
    tight = (48, 24, 10)
    specs = [
        ("l1", None, 9, None),
        ("k", None, 9, tight),
        ("f25w9", 25, 9, None),
        ("f27w9", 27, 9, None),
        ("f29w9", 29, 9, None),
        ("f25w25", 25, 25, None),
        ("f29w25", 29, 25, None),
        ("kf25", 25, 9, tight),
        ("kf29", 29, 9, tight),
    ]
    screens = []
    built = {}
    for label, freeze, active, keeps in specs:
        model = MergedTail(eng, probe, cas.beta, suffix_ids, cas.budget, freeze=freeze, active=active, keeps=keeps)
        built[label] = model
        screens.append(_screen_one(model, tail, rows, label))
    report = {"screens": screens, "cpu_keep": {"k": [48, 24, 10], "drop": 0.00023, "proxy_new_finalists": 0}}
    faithful = [
        s for s in screens
        if s["ms"] <= 1800 and s["top10_overlap_vs_tail"] >= (0.85 if s["freeze"] is None else 0.9)
    ]
    best = min(faithful, key=lambda s: s["ms"]) if faithful else None
    report["pick"] = None if best is None else best["label"]
    path = DATA / "week3_fast_tail.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    if best is None or best["ms"] > 1800:
        report["check"] = "skipped"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return
    ref = torch.load(DATA / "runs/check/collect_4b_b8000.pt", map_location="cpu", weights_only=False)
    report["check"] = _check(built[best["label"]], rows, ref, best["label"])
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def _write_new(name: str, report: dict) -> None:
    path = DATA / name
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def check_frozen(eng, cas, suffix_ids, freeze: int, keeps: tuple[int, int, int] | None, dest: str) -> None:
    """Merged tail, suffix-only after `freeze`, one keep tuple. Writes a new json."""
    probe = load_saved_probe(eng, suffix_ids, "q4_p1")
    rows = torch.load(DATA / "check.pt", weights_only=False)
    ref = torch.load(DATA / "runs/check/collect_4b_b8000.pt", map_location="cpu", weights_only=False)
    model = MergedTail(eng, probe, cas.beta, suffix_ids, cas.budget, freeze=freeze, active=9, keeps=keeps)
    label = dest.replace(".json", "")
    report = _check(model, rows, ref, label)
    report["freeze"] = freeze
    report["active"] = 9
    report["keeps"] = [60, 40, 15] if keeps is None else list(keeps)
    _write_new(dest, report)


def screen_k15(eng, cas, suffix_ids) -> None:
    """Time K21 in {14, 15} under the layer-25 freeze, then check the fastest on the 400."""
    probe = load_saved_probe(eng, suffix_ids, "q4_p1")
    tail = TailCatch(eng, probe, cas.beta, suffix_ids, cas.budget)
    rows = torch.load(DATA / "check.pt", weights_only=False)
    specs = [
        ("48/24/15", 25, (48, 24, 15)),
        ("48/28/15", 25, (48, 28, 15)),
        ("50/28/14", 25, (50, 28, 14)),
        ("60/40/15@23", 23, None),
    ]
    screens = []
    built = {}
    for label, freeze, keeps in specs:
        model = MergedTail(eng, probe, cas.beta, suffix_ids, cas.budget, freeze=freeze, active=9, keeps=keeps)
        built[label] = (model, freeze, keeps)
        screens.append(_screen_one(model, tail, rows, label))
    best = min(screens, key=lambda s: s["ms"])
    report = {"screens": screens, "pick": best["label"], "under_1600": best["ms"] <= 1600}
    _write_new("week3_f25_k15_screen.json", report)
    _model, freeze, keeps = built[best["label"]]
    k9, k15, k21 = (60, 40, 15) if keeps is None else keeps
    dest = f"week3_f{freeze}_{k9}_{k15}_{k21}.json"
    ref = torch.load(DATA / "runs/check/collect_4b_b8000.pt", map_location="cpu", weights_only=False)
    checked = _check(_model, rows, ref, best["label"])
    checked["freeze"] = freeze
    checked["active"] = 9
    checked["keeps"] = [k9, k15, k21]
    checked["screen_ms"] = best["ms"]
    _write_new(dest, checked)


def check_k(eng, cas, suffix_ids) -> None:
    """The (48, 24, 10) keep with the merged tail. No document freeze."""
    probe = load_saved_probe(eng, suffix_ids, "q4_p1")
    rows = torch.load(DATA / "check.pt", weights_only=False)
    ref = torch.load(DATA / "runs/check/collect_4b_b8000.pt", map_location="cpu", weights_only=False)
    model = MergedTail(eng, probe, cas.beta, suffix_ids, cas.budget, keeps=(48, 24, 10))
    report = _check(model, rows, ref, "k")
    (DATA / "week3_fast_k_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    core.FLEX_ON = False
    torch.set_grad_enabled(False)
    eng, cas, suffix_ids = load_all()
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode == "collect":
        collect(eng, cas, suffix_ids)
    elif mode == "check-k":
        check_k(eng, cas, suffix_ids)
    elif mode == "check-f25":
        check_frozen(eng, cas, suffix_ids, 25, None, "week3_f25_60_40_15.json")
    elif mode == "check-keep":
        check_frozen(
            eng, cas, suffix_ids, int(sys.argv[2]),
            (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])), sys.argv[6],
        )
    elif mode == "screen-k15":
        screen_k15(eng, cas, suffix_ids)
    else:
        screen(eng, cas, suffix_ids)


if __name__ == "__main__":
    main()
