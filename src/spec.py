#!/usr/bin/env python3
"""Self-speculative reranking, evaluated from saved per-layer last-token states.

Every document runs to layer l. A head on the layer-l last token ranks the 100.
Only the top K (or an adaptive set) continue to the last layer and get the
unchanged model's score; the rest are ranked below them by the head score.
Heads are fit on the fit split only; every number is reported on the pilot.

Compute is exact per document: a document that stops after layer l costs
(l+1)/L of its dense MACs, a kept one costs all of them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from fit_rules import ndcg, order_ids  # noqa: E402
from train_qwen_draft import OUT  # noqa: E402

DATA = OUT / "fast06"
L = 28


def load(split: str):
    run = torch.load(DATA / "runs" / split / "collect_fp32.pt")
    hid = torch.load(DATA / "runs" / split / "collect_fp32_hidden.pt")
    rows = {r["qid"]: r for r in torch.load(DATA / f"{split}.pt")}
    qids = [q for q in hid["hidden"] if q in rows]
    return run, hid, rows, qids


def rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def features(hid, qids, k: int) -> torch.Tensor:
    return torch.stack([rms(hid["hidden"][q][:, k].float()) for q in qids])


def fit_ridge(X: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """Within-query ridge: X (Q,100,d), y (Q,100). Centered per query, so only order inside a query is fit."""
    Xc = (X - X.mean(1, keepdim=True)).reshape(-1, X.shape[-1]).double()
    yc = (y - y.mean(1, keepdim=True)).reshape(-1).double()
    A = Xc.T @ Xc + lam * Xc.shape[0] * torch.eye(Xc.shape[1], dtype=torch.float64)
    return torch.linalg.solve(A, Xc.T @ yc).float()


def lens_score(X: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (X * gamma) @ w


def evaluate(rows, run, qids, stages) -> dict:
    """stages: list of (layer, head (Q,100), keep_fn); each stage filters the survivors of the last."""
    qrels, orders, refo = {}, {}, {}
    cost, dense_tot, cover, jac, kept_n = 0.0, 0.0, [], [], []
    for qi, q in enumerate(qids):
        r = rows[q]
        ref = run["queries"][q]["margins"]
        dense = run["queries"][q]["dense"]
        alive = list(range(100))
        tiers = []
        stop = {}
        for layer, head, keep_fn in stages:
            h = head[qi].numpy()
            sub = np.array([h[i] for i in alive])
            keep = {alive[j] for j in keep_fn(sub)}
            dropped = [i for i in alive if i not in keep]
            for i in dropped:
                stop[i] = layer
            tiers.append(sorted(dropped, key=lambda i: (h[i], r["ids"][i]), reverse=True))
            alive = [i for i in alive if i in keep]
        kept_n.append(len(alive))
        for i, d in enumerate(dense):
            cost += d * (stop[i] + 1) / L if i in stop else d
            dense_tot += d
        top_ref = sorted(range(100), key=lambda i: (ref[i], r["ids"][i]), reverse=True)[:10]
        cover.append(float(set(top_ref) <= set(alive)))
        final_idx = sorted(alive, key=lambda i: (ref[i], r["ids"][i]), reverse=True)
        for tier in reversed(tiers):
            final_idx += tier
        final = [r["ids"][i] for i in final_idx]
        orders[q] = final
        refo[q] = order_ids({d: m for d, m in zip(r["ids"], ref)})
        jac.append(len(set(final[:10]) & set(refo[q][:10])) / len(set(final[:10]) | set(refo[q][:10])))
        if r["qrels"]:
            qrels[q] = r["qrels"]
    n_ref = ndcg(qrels, {k: v for k, v in refo.items() if k in qrels})
    n_run = ndcg(qrels, {k: v for k, v in orders.items() if k in qrels})
    return {
        "stages": [(s[0]) for s in stages],
        "mac_frac": cost / dense_tot,
        "ndcg@10": n_run,
        "drop": n_ref - n_run,
        "jaccard@10": float(np.mean(jac)),
        "cover": float(np.mean(cover)),
        "mean_kept": float(np.mean(kept_n)),
    }


def top_k(k: int):
    return lambda h: list(np.argsort(-h, kind="stable")[:k])


def adaptive(delta: float, kmin: int, kmax: int):
    def fn(h):
        order = np.argsort(-h, kind="stable")
        cut = h[order[9]] - delta
        keep = [i for i in order if h[i] >= cut][:kmax]
        return keep if len(keep) >= kmin else list(order[:kmin])
    return fn


def main() -> None:
    from transformers import AutoModelForCausalLM

    pins = json.loads((OUT / "manifest.json").read_text())["pins"]
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", dtype=torch.float32)
    wt = model.lm_head.weight
    w = (wt[pins["yes_id"]] - wt[pins["no_id"]]).detach().float()
    gamma = model.model.norm.weight.detach().float()
    del model

    frun, fhid, frows, fq = load("fit")
    prun, phid, prows, pq = load("pilot")
    layers = fhid["layers"]
    yf = torch.tensor([frun["queries"][q]["margins"] for q in fq])
    results = {"single": [], "adaptive": [], "two_stage": []}
    heads_pilot = {}
    for k, layer in enumerate(layers):
        Xf = features(fhid, fq, k)
        Xp = features(phid, pq, k)
        best = None
        for lam in (1e-4, 1e-3, 1e-2):
            beta = fit_ridge(Xf[: int(0.8 * len(fq))], yf[: int(0.8 * len(fq))], lam)
            Xv, yv = Xf[int(0.8 * len(fq)) :], yf[int(0.8 * len(fq)) :]
            pred = Xv @ beta
            rank_corr = float(np.mean([np.corrcoef(pred[i].numpy(), yv[i].numpy())[0, 1] for i in range(len(yv))]))
            if best is None or rank_corr > best[0]:
                best = (rank_corr, lam)
        beta = fit_ridge(Xf, yf, best[1])
        hp = Xp @ beta
        heads_pilot[layer] = hp
        lens = lens_score(Xp, gamma, w)
        for name, head in (("ridge", hp), ("lens", lens)):
            for K in (10, 15, 20, 30, 40, 60):
                res = evaluate(prows, prun, pq, [(layer, head, top_k(K))])
                res.update({"head": name, "K": K, "lam": best[1], "val_corr": best[0]})
                results["single"].append(res)
        print(f"layer {layer} ridge val within-query corr {best[0]:.3f} lam {best[1]}", flush=True)
        for delta in (0.5, 1.0, 2.0, 3.0, 4.0):
            res = evaluate(prows, prun, pq, [(layer, hp, adaptive(delta, 10, 100))])
            res.update({"head": "ridge", "delta": delta})
            results["adaptive"].append(res)

    for l1 in (4, 6, 8):
        for l2 in (12, 14, 16, 18):
            for k1 in (30, 40, 60):
                for k2 in (12, 15, 20):
                    stages = [(l1, heads_pilot[l1], top_k(k1)), (l2, heads_pilot[l2], top_k(k2))]
                    res = evaluate(prows, prun, pq, stages)
                    res.update({"K": [k1, k2]})
                    results["two_stage"].append(res)
    torch.save({"qids": pq, "heads": heads_pilot}, DATA / "spec_pilot_heads.pt")
    results["three_stage"] = []
    for l1, l2, l3 in ((6, 10, 14), (8, 12, 16), (8, 12, 14), (6, 12, 16), (10, 12, 14), (10, 14, 18)):
        for k1 in (40, 60, 80):
            for k2 in (25, 35):
                for k3 in (10, 12, 15, 20):
                    stages = [(l1, heads_pilot[l1], top_k(k1)), (l2, heads_pilot[l2], top_k(k2)), (l3, heads_pilot[l3], top_k(k3))]
                    res = evaluate(prows, prun, pq, stages)
                    res.update({"K": [k1, k2, k3]})
                    results["three_stage"].append(res)
    (DATA / "spec_results.json").write_text(json.dumps(results, indent=1))
    print(f"{'head':6s} {'layer':>5s} {'K':>4s} {'MACs':>6s} {'nDCG':>7s} {'drop':>7s} {'J@10':>6s} {'cover':>6s}")
    for r in sorted(results["single"], key=lambda r: r["mac_frac"]):
        if r["drop"] <= 0.01:
            print(f"{r['head']:6s} {r['stages'][0]:5d} {r['K']:4d} {r['mac_frac']:6.3f} {r['ndcg@10']:7.4f} {r['drop']:+7.4f} {r['jaccard@10']:6.3f} {r['cover']:6.3f}")
    print("adaptive")
    for r in sorted(results["adaptive"], key=lambda r: r["mac_frac"]):
        if r["drop"] <= 0.01:
            print(f"ridge  {r['stages'][0]:5d} d{r['delta']:<3g} {r['mac_frac']:6.3f} {r['ndcg@10']:7.4f} {r['drop']:+7.4f} {r['jaccard@10']:6.3f} {r['cover']:6.3f} kept {r['mean_kept']:.1f}")

    print("multi-stage")
    for r in sorted(results["two_stage"] + results["three_stage"], key=lambda r: r["mac_frac"])[:400]:
        if r["drop"] <= 0.01:
            print(f"{r['stages']} K{r['K']} {r['mac_frac']:6.3f} {r['ndcg@10']:7.4f} {r['drop']:+7.4f} {r['jaccard@10']:6.3f} {r['cover']:6.3f}")


if __name__ == "__main__":
    main()
