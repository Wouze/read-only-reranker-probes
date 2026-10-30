#!/usr/bin/env python3
"""Trained probe tokens: a speculative draft that reads the exact document cache.

Document tokens run the unchanged model through layers 0..l (no gradients, so
their states and K/V are exactly the dense ones and survivors reuse them).
A few probe tokens are appended after the document. They start from the
suffix embeddings, pass through the same layers with LoRA on every linear,
attend to prefix + document K/V, and never write anything back. An exit
head on the probe's last token at layer l predicts the full model's margin.

Usage:
  probe.py train --exits 4,6,8,10,12 [--trunc N] [--epochs 2] [--tag NAME]
  probe.py score --weights PATH --split pilot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from core import Manual, Plan  # noqa: E402
from train_qwen_draft import OUT  # noqa: E402

DATA = OUT / "fast06"
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
BF16 = False


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        object.__setattr__(self, "base", base)
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T) @ self.B.T * self.scale


class LayerLoRA(nn.Module):
    def __init__(self, layer, r: int, alpha: float) -> None:
        super().__init__()
        a, m = layer.self_attn, layer.mlp
        self.q, self.k, self.v, self.o = (LoRA(p, r, alpha) for p in (a.q_proj, a.k_proj, a.v_proj, a.o_proj))
        self.gate, self.up, self.down = (LoRA(p, r, alpha) for p in (m.gate_proj, m.up_proj, m.down_proj))


class ExitHead(nn.Module):
    def __init__(self, gamma: torch.Tensor, w: torch.Tensor) -> None:
        super().__init__()
        self.g = nn.Parameter(gamma.detach().float().clone())
        self.w = nn.Parameter(w.detach().float().clone())

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (x * self.g) @ self.w


class Probe(nn.Module):
    def __init__(self, eng: Manual, suffix_ids: list[int], exits: list[int], r: int = 16, alpha: float = 32) -> None:
        super().__init__()
        self.exits = sorted(exits)
        emb = eng.base.embed_tokens(torch.tensor(suffix_ids, device=eng.device)).detach().float()
        self.emb = nn.Parameter(emb.clone())
        self.ad = nn.ModuleList(LayerLoRA(eng.base.layers[j], r, alpha) for j in range(self.exits[-1] + 1))
        self.heads = nn.ModuleDict({str(e): ExitHead(eng.base.norm.weight, eng.w) for e in self.exits})
        self.doc_depth = None

    def forward(self, eng: Manual, cache, P: int, kvs, lens, valid, frozen=None, handoff_h=None):
        B, S = lens.shape[0], self.emb.shape[0]
        hq, hd = eng.hq, eng.hd
        start_layer = int(getattr(self, "start_layer", 0) or 0)
        x = self.emb[None].expand(B, S, -1)
        if getattr(self, "handoff", False) and start_layer and handoff_h is None:
            raise RuntimeError("handoff probe missing the document state")
        if handoff_h is not None and start_layer:
            x = handoff_h.float()[:, None, :] + self.emb[None]
        pos = P + lens[:, None] + torch.arange(S, device=lens.device)
        cos, sin = eng.base.rotary_emb(x, pos)
        W = valid.shape[1]
        mask = torch.cat([
            torch.ones((B, S, P), dtype=torch.bool, device=lens.device),
            valid[:, None, :].expand(B, S, W),
            torch.ones((S, S), dtype=torch.bool, device=lens.device).tril()[None].expand(B, S, S),
        ], 2)
        out = {}
        for j, ad in enumerate(self.ad):
            if j < start_layer:
                continue
            layer = eng.base.layers[j]
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            q = attn.q_norm(ad.q(h).view(B, S, hq, hd)).transpose(1, 2)
            k = attn.k_norm(ad.k(h).view(B, S, -1, hd)).transpose(1, 2)
            v = ad.v(h).view(B, S, -1, hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            pk, pv = cache[j]
            if j < len(kvs):
                dk, dv = kvs[j]
            else:
                xs, dcos, dsin = frozen
                hs = layer.input_layernorm(xs)
                dk = attn.k_norm(ad.k(hs).view(B, W, -1, hd)).transpose(1, 2)
                dv = ad.v(hs).view(B, W, -1, hd).transpose(1, 2)
                dk, _ = apply_rotary_pos_emb(dk, dk, dcos, dsin)
            dt = q.dtype
            kf = torch.cat([pk.expand(B, -1, -1, -1).to(dt), dk.to(dt), k.to(dt)], 2)
            vf = torch.cat([pv.expand(B, -1, -1, -1).to(dt), dv.to(dt), v.to(dt)], 2)
            a = F.scaled_dot_product_attention(q, kf, vf, attn_mask=mask[:, None], enable_gqa=True, scale=attn.scaling)
            x = x + ad.o(a.transpose(1, 2).reshape(B, S, -1))
            h2 = layer.post_attention_layernorm(x)
            x = x + ad.down(F.silu(ad.gate(h2)) * ad.up(h2))
            if j in self.exits:
                out[j] = self.heads[str(j)](x[:, -1])
        return out

    def trainable(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if not k.endswith(".base")}


@torch.no_grad()
def doc_kv(eng: Manual, cache, P: int, docs: list[list[int]], upto: int, depth=None, capture_at: int | None = None):
    """Exact dense K/V of document tokens (no suffix) for layers 0..upto.

    With depth s < upto, document tokens stop after layer s: K/V for layers
    0..s are exact, and the layer-s output is returned for the probe's own
    K/V projections in deeper layers.
    """
    B = len(docs)
    W = max(1, max(len(d) for d in docs))
    dev = eng.device
    ids = torch.zeros((B, W), dtype=torch.long, device=dev)
    lens = torch.tensor([len(d) for d in docs], device=dev)
    for i, d in enumerate(docs):
        if d:
            ids[i, : len(d)] = torch.tensor(d, device=dev)
    t = torch.arange(W, device=dev)
    valid = t[None] < lens[:, None]
    x = eng.base.embed_tokens(ids)
    cos, sin = eng.base.rotary_emb(x, (P + t)[None].expand(B, W))
    mask = torch.cat([torch.ones((B, W, P), dtype=torch.bool, device=dev),
                      torch.ones((W, W), dtype=torch.bool, device=dev).tril()[None] & valid[:, None, :]], 2)
    plan = Plan()
    kvs = []
    captured = None
    last = upto if depth is None else min(depth, upto)
    for j in range(last + 1):
        if capture_at is not None and j == capture_at:
            rows = torch.arange(B, device=dev)
            captured = x[rows, (lens - 1).clamp(min=0)].detach()
        layer = eng.base.layers[j]
        q, k, v = eng._proj(layer, layer.input_layernorm(x), plan)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        kvs.append((k, v))
        if j == upto:
            break
        pk, pv = cache[j]
        kf = torch.cat([pk.expand(B, -1, -1, -1), k], 2)
        vf = torch.cat([pv.expand(B, -1, -1, -1), v], 2)
        a = F.scaled_dot_product_attention(q, kf, vf, attn_mask=mask[:, None], enable_gqa=True, scale=layer.self_attn.scaling)
        x = x + layer.self_attn.o_proj(a.transpose(1, 2).reshape(B, W, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    if last < upto:
        return kvs, lens, valid, (x, cos, sin), captured
    return kvs, lens, valid, None, captured


def doc_chunks(eng, probe, row, trunc: int, budget: int, amp: bool = False):
    """Prefix cache and exact document K/V per length-sorted chunk of one query."""
    S = row["suffix_len"]
    docs = [c[:-S][:trunc] if trunc else c[:-S] for c in row["conts"]]
    cache, P = eng.prefix(row["prefix"], Plan())
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
    chunks = []
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * max(1, len(docs[order[j]])) <= budget:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            capture = probe.start_layer if getattr(probe, "handoff", False) else None
            kvs, lens, valid, frozen, captured = doc_kv(
                eng, cache, P, [docs[k] for k in idx], probe.exits[-1], probe.doc_depth, capture)
        if not BF16:
            kvs = [(k.float(), v.float()) for k, v in kvs]
            if frozen is not None:
                frozen = tuple(t.float() for t in frozen)
        chunks.append((idx, kvs, lens, valid, frozen, captured))
        i = j
    return cache, P, chunks, len(docs)


def run_chunk(eng, probe, cache, P, chunk):
    idx, kvs, lens, valid, frozen = chunk[:5]
    hand = chunk[5] if len(chunk) > 5 else None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
        return probe(eng, cache, P, kvs, lens, valid, frozen, handoff_h=hand)


def query_scores(eng, probe, row, trunc: int, budget: int, amp: bool = False):
    """Probe margins per exit for all docs of one query, shape {exit: (n_docs,)}, in original doc order."""
    cache, P, chunks, n = doc_chunks(eng, probe, row, trunc, budget, amp)
    parts = {e: [None] * n for e in probe.exits}
    for chunk in chunks:
        for e, m in run_chunk(eng, probe, cache, P, chunk).items():
            for k, val in zip(chunk[0], m):
                parts[e][k] = val
    return {e: torch.stack(v) for e, v in parts.items()}


def load_engine(model_id: str = MODEL_ID, bf16: bool = False):
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    pins = json.loads((OUT / "manifest.json").read_text())["pins"]
    return Manual(model, pins["yes_id"], pins["no_id"])


def score_split(eng, probe, split: str, trunc: int, budget: int, limit: int = 0, qids=None) -> dict:
    rows = torch.load(DATA / f"{split}.pt")
    if qids is not None:
        want = set(qids)
        rows = [r for r in rows if r["qid"] in want]
        if len(rows) != len(want):
            raise RuntimeError(f"score filter kept {len(rows)} of {len(want)}")
    elif limit:
        rows = rows[:limit]
    probe.eval()
    out = {"qids": [], "heads": {e: [] for e in probe.exits}}
    with torch.no_grad():
        for n, row in enumerate(rows, 1):
            s = query_scores(eng, probe, row, trunc, budget)
            out["qids"].append(row["qid"])
            for e in probe.exits:
                out["heads"][e].append(s[e].float().cpu())
            if n % 20 == 0 or n == len(rows):
                print(f"score {split} {n}/{len(rows)}", flush=True)
    out["heads"] = {e: torch.stack(v) for e, v in out["heads"].items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "score"])
    ap.add_argument("--exits", default="4,6,8,10,12")
    ap.add_argument("--trunc", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--bf16", action="store_true", help="bf16 weights, probe path under autocast")
    ap.add_argument("--teacher", default="ref_fp32", help="runs/fit/<name>.pt margins")
    ap.add_argument("--doc_depth", type=int, default=-1, help="document tokens stop after this layer (-1: never)")
    ap.add_argument("--probe_len", type=int, default=0, help="probe tokens = last k suffix tokens (0: whole suffix)")
    ap.add_argument("--probe_start", type=int, default=0, help="probe tokens enter at this layer")
    ap.add_argument("--handoff", action="store_true", help="initialize a late probe from the document last-token state")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for the frozen document pass while training")
    ap.add_argument("--teacher_only", action="store_true", help="score only queries in the teacher file")
    ap.add_argument("--out", default="", help="score output path")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    global BF16
    if args.mode == "score":
        saved = torch.load(args.weights, map_location="cpu")["args"]
        args.model, args.bf16 = saved.get("model", MODEL_ID), saved.get("bf16", False)
    BF16 = args.bf16
    eng = load_engine(args.model, args.bf16)
    rows0 = torch.load(DATA / "pilot.pt")
    S = rows0[0]["suffix_len"]
    suffix_ids = rows0[0]["conts"][0][-S:]
    exits = [int(e) for e in args.exits.split(",")]
    if args.mode == "score":
        args.probe_len = torch.load(args.weights, map_location="cpu")["args"].get("probe_len", 0)
    if args.probe_len:
        suffix_ids = suffix_ids[-args.probe_len:]
    probe = Probe(eng, suffix_ids, exits, r=args.rank).to(eng.device)
    probe.doc_depth = None if args.doc_depth < 0 else args.doc_depth
    probe.start_layer = args.probe_start
    probe.handoff = bool(args.handoff)
    for j, ad in enumerate(probe.ad):
        if j < probe.start_layer:
            for prm in ad.parameters():
                prm.requires_grad_(False)
    pdir = DATA / "probes"
    pdir.mkdir(exist_ok=True)

    if args.mode == "score":
        state = torch.load(args.weights, map_location=eng.device)
        probe = Probe(eng, suffix_ids, state["args"]["exits"], r=state["args"]["rank"]).to(eng.device)
        dd = state["args"].get("doc_depth", -1)
        probe.doc_depth = None if dd < 0 else dd
        probe.start_layer = int(state["args"].get("probe_start", 0) or 0)
        probe.handoff = bool(state["args"].get("handoff", False))
        probe.load_state_dict(state["state"], strict=False)
        qids = None
        if args.teacher_only:
            teacher = state["args"].get("teacher", "collect_4b_b8000")
            ref = torch.load(DATA / "runs" / "fit" / f"{teacher}.pt", map_location="cpu")
            qids = list(ref["queries"])
            print(f"teacher_only {len(qids)} from {teacher}", flush=True)
        res = score_split(eng, probe, args.split, state["args"]["trunc"], args.budget, qids=qids)
        dest = Path(args.out) if args.out else pdir / f"{Path(args.weights).stem}_{args.split}.pt"
        torch.save(res, dest)
        print(f"wrote {dest}", flush=True)
        return

    rows = torch.load(DATA / "fit.pt")
    ref = torch.load(DATA / "runs" / "fit" / f"{args.teacher}.pt")
    rows = [r for r in rows if r["qid"] in ref["queries"]]
    params = [p for p in probe.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.2f}M, {len(rows)} fit queries, exits {exits}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = int(args.epochs * len(rows))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    rng = random.Random(0)
    t0 = time.time()
    run_loss = {e: 0.0 for e in exits}
    step = 0
    while step < steps:
        rng.shuffle(rows)
        for row in rows:
            if step >= steps:
                break
            probe.train()
            cache, P, chunks, n = doc_chunks(eng, probe, row, args.trunc, args.budget, args.amp)
            with torch.no_grad():
                parts = {e: torch.zeros(n, device=eng.device) for e in exits}
                for chunk in chunks:
                    for e, m in run_chunk(eng, probe, cache, P, chunk).items():
                        parts[e][torch.tensor(chunk[0], device=eng.device)] = m.float()
            t = torch.tensor(ref["queries"][row["qid"]]["margins"], device=eng.device)
            tc = t - t.mean()
            grads = {}
            for e, m0 in parts.items():
                m = m0.clone().requires_grad_(True)
                mc = m - m.mean()
                le = (mc - tc).pow(2).mean() + F.kl_div(torch.log_softmax(m / args.tau, 0), torch.softmax(t / args.tau, 0), reduction="sum")
                le.backward()
                grads[e] = m.grad
                run_loss[e] += float(le.detach())
            opt.zero_grad()
            for chunk in chunks:
                sel = torch.tensor(chunk[0], device=eng.device)
                out = run_chunk(eng, probe, cache, P, chunk)
                sum((out[e].float() * grads[e][sel]).sum() for e in exits).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{steps} {time.time() - t0:.0f}s " + " ".join(f"L{e}:{run_loss[e] / 100:.3f}" for e in exits), flush=True)
                run_loss = {e: 0.0 for e in exits}
    name = args.tag or f"probe_e{'-'.join(map(str, exits))}_t{args.trunc}"
    dest = pdir / f"{name}.pt"
    torch.save({"state": probe.trainable(), "args": {**vars(args), "exits": exits}}, dest)
    print(f"saved {dest} {time.time() - t0:.0f}s", flush=True)
    res = score_split(eng, probe, "pilot", args.trunc, args.budget)
    torch.save(res, pdir / f"{name}_pilot.pt")
    print(f"scored pilot {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
