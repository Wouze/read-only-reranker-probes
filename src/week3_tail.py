#!/usr/bin/env python3
"""Head-192 through the probes, then the tail is caught up before the ridge.

Layers 0-15 and both probes see only the first 192 document tokens. The 40
survivors then run doc[192:] through those same layers, and the ridge and the
final margin see the full passage. One check-split pass if the pilot margin
is not nonsense. Does not refit and does not open the holdout, dev, or DL19.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import core
from core import Plan
from fit_rules import ndcg
from next_exps import Catchup, ContCascade, probe_forward, slice_rows
from paper_eval import DATA, advance, batch_state, chunks, jaccard, keep_top, load_all, timed
from week_exps import _stitch, load_saved_probe, tail_through

CAP = 192


class TailCatch(Catchup):
    def run(self, row: dict):
        eng = self.eng
        S = len(self.suffix_ids)
        docs = [c[:-S] for c in row["conts"]]
        heads = [d[:CAP] for d in docs]
        n = len(docs)
        cache, P = eng.prefix(row["prefix"], Plan())
        self._probe_x = {}
        state = {}
        stage_score = torch.zeros(n)
        for idx in chunks([max(1, len(d)) for d in heads], self.budget):
            x, kvs, lens, valid = advance(eng, cache, P, [heads[i] for i in idx], None, None, 0, self.stages[0][0])
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
                state.pop(i, None)
        depth = self.stages[0][0]
        for tier_i, (layer, k) in enumerate(self.stages[1:], start=2):
            alive, stage_score = ContCascade._probe_stage(
                self, cache, P, heads, state, alive, depth, layer, k, stage_score
            )
            for i in list(state):
                if i not in set(alive):
                    if tier_key[i] == 0:
                        tier_key[i] = tier_i
                        tier_score[i] = stage_score[i]
                    self._probe_x.pop(i, None)
                    state.pop(i, None)
            depth = layer
        self._long = sum(1 for i in alive if len(docs[i]) > CAP)
        state = self._catch_tails(cache, P, docs, state, alive, depth)
        return self._finish(cache, P, row["conts"], alive, tier_key, tier_score, state)

    def _catch_tails(self, cache, P, docs, state, alive, end):
        eng = self.eng
        if not alive:
            return state
        out = {}
        for idx in chunks([len(docs[alive[i]]) for i in range(len(alive))], self.budget):
            chosen = [alive[i] for i in idx]
            tails = [docs[i][state[i]["x"].shape[0] :] for i in chosen]
            if any(tails):
                px, pkvs, plens = batch_state([(i, state[i]["x"], state[i]["kvs"]) for i in chosen])
                pre = torch.tensor(plens, device=eng.device)
                tx, tkvs, tlens = tail_through(eng, cache, P, pkvs, pre, tails, end)
                xj, kvsj, lensj = _stitch(px, pkvs, pre, tx, tkvs, tlens)
                for local, doc_i in enumerate(chosen):
                    ntok = int(lensj[local])
                    out[doc_i] = {
                        "x": xj[local, :ntok].contiguous(),
                        "kvs": [(kk[local, :, :ntok].contiguous(), vv[local, :, :ntok].contiguous()) for kk, vv in kvsj],
                    }
            else:
                for doc_i in chosen:
                    out[doc_i] = state[doc_i]
        return out


def _gap(exact_a, exact_b):
    both = [
        i
        for i in range(len(exact_a))
        if float(exact_a[i]) > float("-inf") and float(exact_b[i]) > float("-inf")
    ]
    if not both:
        return None
    return max(abs(float(exact_a[i]) - float(exact_b[i])) for i in both)


def main() -> None:
    core.FLEX_ON = False
    torch.set_grad_enabled(False)
    eng, cas, suffix_ids = load_all()
    probe = load_saved_probe(eng, suffix_ids, "q4_p1")
    full = Catchup(eng, probe, cas.beta, suffix_ids, cas.budget)
    tail = TailCatch(eng, probe, cas.beta, suffix_ids, cas.budget)
    pilot = torch.load(DATA / "pilot.pt")
    for row in pilot[:2]:
        full.run(row)
        tail.run(row)
    base_ms, tail_ms, gaps, longs = [], [], [], []
    for row in pilot[2:10]:
        base_ms.append(timed(lambda row=row: full.run(row)))
        _ob, eb, _k, _s = full.run(row)
        tail_ms.append(timed(lambda row=row: tail.run(row)))
        _oa, ea, _k, _s = tail.run(row)
        gaps.append(_gap(eb, ea))
        longs.append(tail._long)
        print(
            f"pilot {row['qid']} full {base_ms[-1]:.0f} tail {tail_ms[-1]:.0f} "
            f"gap {gaps[-1]} long {longs[-1]}",
            flush=True,
        )
    pilot_report = {
        "n": len(base_ms),
        "full_ms": sum(base_ms) / len(base_ms),
        "tail_ms": sum(tail_ms) / len(tail_ms),
        "finalist_margin_max_abs_vs_full": max(g for g in gaps if g is not None),
        "mean_long_survivors": sum(longs) / len(longs),
    }
    print(json.dumps(pilot_report), flush=True)
    if pilot_report["finalist_margin_max_abs_vs_full"] > 2:
        pilot_report["check"] = "skipped, pilot margin gap above 2"
        (DATA / "week3_tail.json").write_text(json.dumps(pilot_report, indent=2) + "\n")
        return
    rows = torch.load(DATA / "check.pt")
    ref = torch.load(DATA / "runs/check/collect_4b_b8000.pt", map_location="cpu")
    for row in rows[:2]:
        tail.run(row)
    times = []
    for row in rows[2:10]:
        times.append(timed(lambda row=row: tail.run(row)))
    jac, cover = [], []
    qrels, got, teacher = {}, {}, {}
    for n, row in enumerate(rows, 1):
        order, exact, _k, _s = tail.run(row)
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
            print(f"check {n}/{len(rows)} J {sum(jac)/len(jac):.3f} cover {sum(cover)/len(cover):.3f}", flush=True)
    teacher_nd = ndcg(qrels, teacher)
    cascade_nd = ndcg(qrels, got)
    report = {
        **pilot_report,
        "check_n": len(rows),
        "check_mean_ms": sum(times) / len(times),
        "jaccard@10": sum(jac) / len(jac),
        "cover": sum(cover) / len(cover),
        "cascade_ndcg@10": cascade_nd,
        "teacher_ndcg@10": teacher_nd,
        "drop": teacher_nd - cascade_nd,
    }
    (DATA / "week3_tail.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
