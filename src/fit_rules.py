#!/usr/bin/env python3
"""Fit H1–H5 on train scores only, then hash rules/h3.json.

Refuses to open a dev score file. Refuses to overwrite a hashed rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("RERANKER_DATA", ROOT / "data"))
RULE = OUT / "rules/h3.json"
RULE_HASH = OUT / "rules/h3.sha256"


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["qid"]] = row
    return rows


def order_ids(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda doc: (scores[doc], doc), reverse=True)


def jaccard(left: list[str], right: list[str], k: int) -> float:
    a, b = set(left[:k]), set(right[:k])
    return len(a & b) / len(a | b)


def ndcg(qrels: dict[str, dict[str, int]], rankings: dict[str, list[str]]) -> float:
    from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores

    results = {}
    for qid, doc_ids in rankings.items():
        n = len(doc_ids)
        results[qid] = {doc_id: float(n - rank) for rank, doc_id in enumerate(doc_ids)}
    clean = {qid: docs for qid, docs in qrels.items() if qid in results}
    results = {qid: docs for qid, docs in results.items() if qid in clean}
    if not results:
        raise SystemExit("no overlapping qrels")
    ev = calculate_retrieval_scores(results, clean, k_values=(10,))
    return float(ev.ndcg["NDCG@10"])


def splice(draft: list[str], teacher_scores: dict[str, float], outside_scores: dict[str, float], band: int) -> list[str]:
    mixed = {}
    head = set(draft[:band])
    for doc in draft:
        mixed[doc] = teacher_scores[doc] if doc in head else outside_scores[doc]
    return order_ids(mixed)


def handoff(draft: list[str], teacher_scores: dict[str, float], band: int) -> list[str]:
    head = draft[:band]
    head_order = sorted(head, key=lambda doc: (teacher_scores[doc], doc), reverse=True)
    return head_order + draft[band:]


def early_yes(margin: float) -> float:
    return float(1.0 / (1.0 + math.exp(margin)))


def main() -> None:
    if RULE.exists() or RULE_HASH.exists():
        raise SystemExit("rules/h3.json already exists; not refitting")
    for banned in OUT.glob("dev_*.jsonl"):
        raise SystemExit(f"dev scores exist before the rule hash: {banned.name}")
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    digest = (OUT / "manifest.sha256").read_text(encoding="utf-8").strip()
    if hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest() != digest:
        raise SystemExit("manifest hash does not match")
    pins = manifest["pins"]
    draft_rows = load_jsonl(OUT / "train_0.6b.jsonl")
    teacher_rows = load_jsonl(OUT / "train_4b.jsonl")
    wanted = [row["id"] for row in manifest["arabic_train_fit"]]
    missing = [qid for qid in wanted if qid not in draft_rows or qid not in teacher_rows]
    if missing:
        raise SystemExit(f"train scores incomplete, missing {len(missing)}")
    qrels = {row["id"]: {item["id"]: int(item["rel"]) for item in row["qrels"]} for row in manifest["arabic_train_fit"]}
    qrels = {qid: docs for qid, docs in qrels.items() if docs}

    ranks: list[int] = []
    draft_orders = {}
    teacher_orders = {}
    for qid in wanted:
        draft_orders[qid] = order_ids(draft_rows[qid]["scores"])
        teacher_orders[qid] = order_ids(teacher_rows[qid]["scores"])
        draft_rank = {doc: i + 1 for i, doc in enumerate(draft_orders[qid])}
        for doc in teacher_orders[qid][:10]:
            ranks.append(draft_rank[doc])
    rank_arr = np.asarray(ranks, dtype=np.float64)
    p95 = float(np.percentile(rank_arr, 95, method="linear"))
    cap = int(pins["h1_rank_cap"])
    if p95 <= cap:
        band = cap
        latency_gate = "half"
    else:
        band = min(100, math.ceil(p95))
        latency_gate = (
            "rewritten before dev: verify the measured train p95 band; "
            "the half-of-full-list latency gate is not claimed"
        )
    buckets = {"1-10": 0, "11-20": 0, "21-35": 0, "36-50": 0, "51-100": 0}
    for rank in ranks:
        if rank <= 10:
            buckets["1-10"] += 1
        elif rank <= 20:
            buckets["11-20"] += 1
        elif rank <= 35:
            buckets["21-35"] += 1
        elif rank <= 50:
            buckets["36-50"] += 1
        else:
            buckets["51-100"] += 1

    xs, ys = [], []
    for qid in wanted:
        draft_scores = draft_rows[qid]["scores"]
        teacher_scores = teacher_rows[qid]["scores"]
        for doc, score in draft_scores.items():
            xs.append(score)
            ys.append(teacher_scores[doc])
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", increasing=True, y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(xs), np.asarray(ys))
    mapped = {}
    for qid in wanted:
        raw = draft_rows[qid]["scores"]
        pred = iso.predict(np.asarray([raw[doc] for doc in raw], dtype=np.float64))
        mapped[qid] = {doc: float(value) for doc, value in zip(raw, pred, strict=True)}

    raw_rankings, mapped_rankings, handoff_rankings = {}, {}, {}
    for qid in wanted:
        raw_rankings[qid] = splice(
            draft_orders[qid], teacher_rows[qid]["scores"], draft_rows[qid]["scores"], band
        )
        mapped_rankings[qid] = splice(
            draft_orders[qid], teacher_rows[qid]["scores"], mapped[qid], band
        )
        handoff_rankings[qid] = handoff(draft_orders[qid], teacher_rows[qid]["scores"], band)
    raw_ndcg = ndcg(qrels, raw_rankings)
    mapped_ndcg = ndcg(qrels, mapped_rankings)
    pair_jaccard = float(np.mean([jaccard(raw_rankings[qid], mapped_rankings[qid], 10) for qid in wanted]))
    map_idle = abs(raw_ndcg - mapped_ndcg) <= 0.005 and pair_jaccard >= 0.99
    splice_rankings = raw_rankings if map_idle else mapped_rankings
    splice_j = float(np.mean([jaccard(splice_rankings[qid], teacher_orders[qid], 10) for qid in wanted]))
    handoff_j = float(np.mean([jaccard(handoff_rankings[qid], teacher_orders[qid], 10) for qid in wanted]))
    primary = "handoff" if handoff_j > splice_j else "band"

    teacher_ndcg = ndcg(qrels, teacher_orders)
    grid = [float(value) for value in pins["h5_margin_grid"]]
    h5_rows = []
    for threshold in grid:
        exited = 0
        total = 0
        rankings = {}
        for qid in wanted:
            margins = teacher_rows[qid].get("early_margin")
            if margins is None:
                raise SystemExit(f"missing early margin on {qid}")
            scores = dict(teacher_rows[qid]["scores"])
            for doc, margin in margins.items():
                total += 1
                if margin >= threshold:
                    scores[doc] = early_yes(margin)
                    exited += 1
            rankings[qid] = order_ids(scores)
        h5_rows.append(
            {
                "margin": threshold,
                "exit_rate": exited / total,
                "ndcg@10": ndcg(qrels, rankings),
            }
        )
    floor = float(pins["h5_exit_rate_floor"])
    eligible = [row for row in h5_rows if row["ndcg@10"] >= teacher_ndcg - 0.005]
    eligible.sort(key=lambda row: (row["exit_rate"], row["margin"]), reverse=True)
    chosen = eligible[0] if eligible else None
    h5_claim = bool(chosen and chosen["exit_rate"] >= floor)

    draft_j10 = float(np.mean([jaccard(draft_orders[qid], teacher_orders[qid], 10) for qid in wanted]))
    rule = {
        "manifest_sha256": digest,
        "n_train": len(wanted),
        "h1": {
            "p50": float(np.percentile(rank_arr, 50, method="linear")),
            "p95": p95,
            "p99": float(np.percentile(rank_arr, 99, method="linear")),
            "buckets": buckets,
            "n_ranks": len(ranks),
            "band": band,
            "latency_gate": latency_gate,
            "draft_jaccard@10": draft_j10,
        },
        "h2": {
            "x_thresholds": [float(v) for v in iso.X_thresholds_],
            "y_thresholds": [float(v) for v in iso.y_thresholds_],
            "raw_ndcg@10": raw_ndcg,
            "mapped_ndcg@10": mapped_ndcg,
            "raw_vs_mapped_jaccard@10": pair_jaccard,
            "map_idle": map_idle,
            "note": "idle when raw and mapped differ by at most 0.005 nDCG@10 and their Jaccard@10 is at least 0.99",
        },
        "h3": {"band": band, "use_map": not map_idle},
        "h4": {
            "splice_jaccard@10": splice_j,
            "handoff_jaccard@10": handoff_j,
            "primary": primary,
            "call_budget": band,
        },
        "h5": {
            "layer_0based": pins["h5_layer_0based"],
            "grid": h5_rows,
            "chosen_margin": None if chosen is None else chosen["margin"],
            "exit_rate": None if chosen is None else chosen["exit_rate"],
            "latency_claim": h5_claim,
            "teacher_ndcg@10": teacher_ndcg,
        },
    }
    RULE.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rule, ensure_ascii=False, indent=2)
    RULE.write_text(text, encoding="utf-8")
    rule_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    RULE_HASH.write_text(rule_digest + "\n", encoding="utf-8")
    print(json.dumps({"sha256": rule_digest, "h1": rule["h1"], "h4": rule["h4"], "h5_claim": h5_claim}, indent=2))


if __name__ == "__main__":
    main()
