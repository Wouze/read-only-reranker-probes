#!/usr/bin/env python3
# After training, a separate scorer must measure the train p95 rank of the 4B
# top 10 under this student. Dev must not be scored unless that p95 is 15 or lower.
# This script does not score dev and does not write a rule file.
"""Fine-tune Qwen/Qwen3-Reranker-0.6B so its yes-probability matches frozen 4B scores.

One epoch on the 2616 arabic_train_fit queries. Dev lists are not read.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("RERANKER_DATA", ROOT / "data"))
DEST = OUT / "draft_qwen"
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
FIT_QUERIES = 2616
PAD_TOL = 1e-3


class LeftPadRejected(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def norm(text: str) -> str:
    table = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
    chars = []
    for ch in (text or "").translate(table):
        if "\u064b" <= ch <= "\u0652" or ch == "\u0640":
            continue
        chars.append(ch)
    return " ".join("".join(chars).split())


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL by physical newline. splitlines() also cuts on U+2028 inside passage text."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_manifest() -> dict:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    digest = (OUT / "manifest.sha256").read_text(encoding="utf-8").strip()
    got = hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest()
    if got != digest:
        raise SystemExit("manifest hash does not match")
    return manifest


def index_qid(rows: list[dict], name: str) -> dict[str, dict]:
    found = {}
    for row in rows:
        qid = row["qid"]
        if qid in found:
            raise SystemExit(f"duplicate qid in {name}: {qid}")
        found[qid] = row
    return found


def check_pins(pins: dict) -> None:
    for key in ("prefix", "suffix", "instruction", "yes_id", "no_id", "max_len"):
        if key not in pins:
            raise SystemExit(f"manifest pins missing {key}")
    if (pins["yes_id"], pins["no_id"], pins["max_len"]) != (9693, 2152, 1024):
        raise SystemExit(
            f"pins yes={pins['yes_id']} no={pins['no_id']} max_len={pins['max_len']} "
            "do not match 9693, 2152, 1024"
        )


def build_pairs(manifest: dict) -> list[tuple[str, str, float]]:
    hold = {row["norm"] for row in manifest["holdout"]}
    lists = index_qid(read_jsonl(OUT / "train_top100.jsonl"), "train_top100.jsonl")
    teacher = index_qid(read_jsonl(OUT / "train_4b.jsonl"), "train_4b.jsonl")
    wanted = [row["id"] for row in manifest["arabic_train_fit"]]
    if len(wanted) != FIT_QUERIES or len(set(wanted)) != FIT_QUERIES:
        raise SystemExit(f"train-fit count {len(wanted)} is not {FIT_QUERIES}")
    missing = [qid for qid in wanted if qid not in teacher or qid not in lists]
    if missing:
        raise SystemExit(f"train 4B scores or top-100 lists are incomplete: {len(missing)}")
    pairs: list[tuple[str, str, float]] = []
    for qid in wanted:
        row = lists[qid]
        if norm(row["query"]) in hold:
            raise SystemExit(f"holdout leaked into draft training: {qid}")
        hits = row["hits"]
        if len(hits) != 100 or len({hit["id"] for hit in hits}) != 100:
            raise SystemExit(f"{qid} does not have 100 passages")
        scores = teacher[qid].get("scores")
        if not isinstance(scores, dict):
            raise SystemExit(f"train 4B row {qid} has no scores")
        for hit in hits:
            try:
                value = float(scores[hit["id"]])
            except KeyError:
                raise SystemExit(f"missing 4B score for {qid} {hit['id']}") from None
            if not math.isfinite(value):
                raise SystemExit(f"non-finite 4B score for {qid} {hit['id']}")
            pairs.append((row["query"], hit["text"], value))
    if len(pairs) != FIT_QUERIES * 100:
        raise SystemExit(f"pair count {len(pairs)} is not {FIT_QUERIES * 100}")
    return pairs


def query_prefix(tok, pins: dict, prefix_ids: list[int], suffix_ids: list[int], query: str) -> list[int]:
    """Pinned prefix and suffix stay whole. The query is cut only when it leaves the document no token."""
    instruct = pins["instruction"]
    fixed = prefix_ids + tok.encode(f"<Instruct>: {instruct}\n<Query>: ", add_special_tokens=False)
    tail = tok.encode("\n<Document>: ", add_special_tokens=False)
    query_ids = tok.encode(query, add_special_tokens=False)
    budget = pins["max_len"] - len(fixed) - len(tail) - len(suffix_ids) - 1
    if budget < 1:
        raise SystemExit("template already fills max_len")
    if len(query_ids) > budget:
        print(
            f"query truncated {len(query_ids)} -> {budget} so the document keeps one token",
            flush=True,
        )
        query_ids = query_ids[:budget]
    return fixed + query_ids + tail


def continuation(tok, pins: dict, suffix_ids: list[int], doc: str, prefix_len: int) -> list[int]:
    """Same document cut as Scorer._continuation when doc_tokens is 0."""
    room = pins["max_len"] - prefix_len - len(suffix_ids)
    if room <= 0:
        raise SystemExit("prefix and suffix already fill max_len")
    doc_ids = tok.encode(doc, add_special_tokens=False)[:room]
    return doc_ids + suffix_ids


def yes_prob_from_logits(logits: torch.Tensor, yes_id: int, no_id: int) -> torch.Tensor:
    yes = logits[:, yes_id]
    no = logits[:, no_id]
    shift = torch.maximum(yes, no)
    ey = torch.exp(yes - shift)
    en = torch.exp(no - shift)
    return ey / (ey + en)


def forward_left(model, seqs: list[list[int]], yes_id: int, no_id: int, pad_id: int) -> torch.Tensor:
    """Left-padded full sequences, same layout as Scorer.score_query_exact."""
    device = model.device
    width = max(len(seq) for seq in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), width), device=device)
    for i, seq in enumerate(seqs):
        ids[i, width - len(seq) :] = torch.tensor(seq, dtype=torch.long, device=device)
        attn[i, width - len(seq) :] = 1
    logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits[:, -1].float()
    return yes_prob_from_logits(logits, yes_id, no_id)


def forward_one(model, seq: list[int], yes_id: int, no_id: int) -> torch.Tensor:
    """One unpadded sequence, same layout as Scorer.full_sequence_prob."""
    logits = model(
        input_ids=torch.tensor([seq], dtype=torch.long, device=model.device),
        use_cache=False,
    ).logits[0, -1].float()
    return yes_prob_from_logits(logits.unsqueeze(0), yes_id, no_id).squeeze(0)


def left_pad_delta(
    model,
    seqs: list[list[int]],
    yes_id: int,
    no_id: int,
    pad_id: int,
    keep_graph: bool,
) -> tuple[torch.Tensor | None, float]:
    """Compare a left-padded batch with each unpadded sequence. Singles run first."""
    with torch.no_grad():
        ref = torch.stack([forward_one(model, seq, yes_id, no_id) for seq in seqs])
    if not keep_graph:
        with torch.no_grad():
            pred = forward_left(model, seqs, yes_id, no_id, pad_id)
            return None, float((pred - ref).abs().max())
    pred = forward_left(model, seqs, yes_id, no_id, pad_id)
    return pred, float((pred.detach() - ref).abs().max())


def first_unequal_batch(pairs, batch: int, encode, start_at: int) -> list[list[int]] | None:
    for start in range(start_at, len(pairs), batch):
        chunk = pairs[start : start + batch]
        if len(chunk) < 2:
            continue
        seqs = [encode(query, doc) for query, doc, _score in chunk]
        if len({len(seq) for seq in seqs}) > 1:
            return seqs
    return None


def train(model, tok, pins: dict, pairs, prefix_ids: list[int], suffix_ids: list[int]) -> tuple[int, int, bool]:
    """One epoch. Left padding is kept only when the first step matches an unpadded forward."""
    yes_id = pins["yes_id"]
    no_id = pins["no_id"]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def encode(query: str, doc: str) -> list[int]:
        prefix = query_prefix(tok, pins, prefix_ids, suffix_ids, query)
        seq = prefix + continuation(tok, pins, suffix_ids, doc, len(prefix))
        if len(seq) > pins["max_len"]:
            raise SystemExit("sequence exceeds max_len")
        return seq

    # Left padding is unsafe when lengths differ. The first step's left-padded
    # yes-probabilities have to match an unpadded single forward within 1e-3.
    # If that batch's lengths are equal, the same check is applied to the first
    # later batch whose lengths differ. Failure trains batch 1 with no padding.
    # CUDA OOM drops the batch 4 -> 2 -> 1.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    batch = 4
    pad_left = True
    checked = False
    index = 0
    step = 0
    model.train()
    print(f"epoch pairs={len(pairs)}", flush=True)
    while index < len(pairs):
        take = batch if pad_left else 1
        chunk = pairs[index : index + take]
        pred = None
        loss = None
        try:
            seqs = [encode(query, doc) for query, doc, _score in chunk]
            target = torch.tensor([item[2] for item in chunk], device=model.device, dtype=torch.float32)
            if pad_left and not checked:
                lengths = [len(seq) for seq in seqs]
                unequal_note = ""
                if len(set(lengths)) < 2:
                    other = first_unequal_batch(pairs, batch, encode, index + len(chunk))
                    if other is None:
                        raise LeftPadRejected("no unequal-length batch")
                    _ignored, extra = left_pad_delta(model, other, yes_id, no_id, pad_id, keep_graph=False)
                    other_lengths = [len(seq) for seq in other]
                    if not math.isfinite(extra) or extra > PAD_TOL:
                        raise LeftPadRejected(
                            f"unequal_delta={extra:.6f} unequal_lengths={other_lengths}"
                        )
                    unequal_note = f" unequal_delta={extra:.6f} unequal_lengths={other_lengths}"
                pred, delta = left_pad_delta(model, seqs, yes_id, no_id, pad_id, keep_graph=True)
                if not math.isfinite(delta) or delta > PAD_TOL:
                    raise LeftPadRejected(f"first_step_delta={delta:.6f} lengths={lengths}")
                print(
                    f"left pad accepted batch {batch} first_step_delta={delta:.6f} "
                    f"lengths={lengths}{unequal_note}",
                    flush=True,
                )
                checked = True
            elif pad_left:
                pred = forward_left(model, seqs, yes_id, no_id, pad_id)
            else:
                pred = forward_one(model, seqs[0], yes_id, no_id).unsqueeze(0)
            loss = torch.mean((pred - target) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        except LeftPadRejected as exc:
            pred = None
            loss = None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(
                f"left pad rejected {exc.detail}; training at batch 1 with no padding",
                flush=True,
            )
            pad_left = False
            batch = 1
            checked = True
            index = 0
            step = 0
            continue
        except torch.OutOfMemoryError:
            pred = None
            loss = None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if batch == 1 or not pad_left:
                raise SystemExit("batch 1 oom")
            prev = batch
            batch = 2 if batch == 4 else 1
            if batch == 1:
                pad_left = False
                if not checked:
                    checked = True
                    print(f"batch {prev} oom; training at batch 1 with no padding", flush=True)
                else:
                    print(f"batch {prev} oom; continuing at batch 1 with no padding", flush=True)
            else:
                print(f"batch {prev} oom", flush=True)
            continue
        value = float(loss.detach())
        pred = None
        loss = None
        index += len(chunk)
        step += 1
        if step % 200 == 0:
            print(f"step {step} loss {value:.4f} batch {len(chunk)}", flush=True)
    return step, batch, pad_left


def main() -> None:
    if (DEST / "config.json").exists():
        print("draft already trained", flush=True)
        return
    manifest = load_manifest()
    pins = manifest["pins"]
    check_pins(pins)
    pairs = build_pairs(manifest)
    random.Random(13).shuffle(pairs)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    yes = tok.convert_tokens_to_ids("yes")
    no = tok.convert_tokens_to_ids("no")
    if yes != pins["yes_id"] or no != pins["no_id"]:
        raise SystemExit(f"token ids yes={yes} no={no} do not match the manifest")
    prefix_ids = tok.encode(pins["prefix"], add_special_tokens=False)
    suffix_ids = tok.encode(pins["suffix"], add_special_tokens=False)
    if suffix_ids.count(tok.convert_tokens_to_ids("<think>")) != 1:
        raise SystemExit("suffix does not contain one think token")
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .train()
    )
    model.config.use_cache = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    step, batch, pad_left = train(model, tok, pins, pairs, prefix_ids, suffix_ids)
    model.save_pretrained(DEST)
    tok.save_pretrained(DEST)
    padding = "left" if pad_left else "none"
    print(f"saved draft pairs={len(pairs)} steps={step} batch={batch} padding={padding}", flush=True)


if __name__ == "__main__":
    main()
