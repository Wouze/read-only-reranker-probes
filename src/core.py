"""Manual Qwen3 reranker forward with per-token depth, pruning, sparsity, and MAC accounting.

The query prefix runs once per query and is cached. Each document continuation
(document tokens + pinned suffix) is scored in a right-padded batch that attends
to that cache. The score is the fp32 yes-minus-no margin of the last token.

Plans change what each continuation token computes. The simulation runs dense
math and discards what a plan drops; MACs are counted analytically from what a
plan keeps. MACs that do not depend on the query (document tokens inside
query-blind layers) are counted as offline, not online.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

inductor_config.max_autotune_gemm_backends = "TRITON,ATEN"
# Flex stays off for the paper forward. Compiling it here holds the process for minutes with no output.
_FLEX = flex_attention
_FLEX_BLOCKS: dict = {}
_FLEX_BAD: set = set()
_FLEX_ERR: dict = {}


FLEX_ON = False


def flex_doc_block(P: int, lens: torch.Tensor, width: int):
    """One block mask for a prefix-cached, left-aligned, right-padded continuation."""
    if not FLEX_ON:
        return None, None
    B = int(lens.shape[0])
    key = (B, width, P + width, tuple(int(v) for v in lens.detach().tolist()))
    if key in _FLEX_BAD:
        return None, key
    block = _FLEX_BLOCKS.get(key)
    if block is None:
        def mask_mod(b, h, qi, ki, P=P, lens=lens):
            return (ki <= qi + P) & (ki < P + lens[b])
        try:
            block = create_block_mask(mask_mod, B, None, width, P + width, device=lens.device)
        except Exception:
            _FLEX_BAD.add(key)
            return None, key
        _FLEX_BLOCKS[key] = block
    return block, key


def flex_doc_attend(q, kf, vf, block, key, scale, attn_mask):
    """Flex when this shape matches the math kernel. Otherwise the math kernel."""
    def math():
        return F.scaled_dot_product_attention(q, kf, vf, attn_mask=attn_mask, enable_gqa=True, scale=scale)
    if block is None or key in _FLEX_BAD:
        return math()
    try:
        out = _FLEX(q, kf, vf, block_mask=block, scale=scale, enable_gqa=True)
    except Exception:
        _FLEX_BAD.add(key)
        return math()
    if key not in _FLEX_ERR:
        err = (out - math()).abs().max().item()
        _FLEX_ERR[key] = err
        if err > 0.05:
            _FLEX_BAD.add(key)
            return math()
    return out


@dataclass
class Plan:
    doc_stop: int | None = None
    prune_layer: int | None = None
    keep: float = 1.0
    keep_min: int = 0
    prune_rows: str = "last"
    skip: tuple[int, ...] = ()
    mlp_density: dict[int, float] = field(default_factory=dict)
    blind_layer: int | None = None
    blind_shift: bool = False
    blind_suffix_doc: bool = True
    x_keep: float = 1.0
    cats_keep: float = 1.0
    mid_keep: float = 1.0
    adapter: str = ""
    collect: tuple[int, ...] = ()
    collect_pool: bool = False
    tag: str = ""

    def name(self) -> str:
        if self.tag:
            return self.tag
        parts = []
        if self.blind_layer is not None:
            parts.append(f"blind{self.blind_layer}" + ("s" if self.blind_shift else "") + ("" if self.blind_suffix_doc else "n"))
        if self.doc_stop is not None:
            parts.append(f"stop{self.doc_stop}")
        if self.prune_layer is not None:
            parts.append(f"prune{self.prune_layer}k{self.keep:g}{self.prune_rows}")
        if self.skip:
            parts.append("skip" + "-".join(map(str, self.skip)))
        if self.x_keep < 1:
            parts.append(f"x{self.x_keep:g}")
        if self.cats_keep < 1:
            parts.append(f"cats{self.cats_keep:g}")
        if self.mid_keep < 1:
            parts.append(f"mid{self.mid_keep:g}")
        if self.adapter:
            parts.append("ad-" + Path(self.adapter).stem)
        if self.collect:
            parts.append("collectpool" if self.collect_pool else "collect")
        return "_".join(parts) or "ref"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["skip"] = list(self.skip)
        out["collect"] = list(self.collect)
        out["mlp_density"] = {str(k): v for k, v in self.mlp_density.items()}
        return out


def topk_mask(t: torch.Tensor, keep: float) -> torch.Tensor:
    if keep >= 1:
        return t
    k = max(1, int(round(keep * t.shape[-1])))
    thresh = t.abs().topk(k, -1).values[..., -1:]
    return t * (t.abs() >= thresh)


class KVAdapter(nn.Module):
    """Key/value projections for frozen document tokens in one layer, initialized from that layer."""

    def __init__(self, layer) -> None:
        super().__init__()
        attn = layer.self_attn
        self.hd = attn.head_dim
        self.norm = copy.deepcopy(layer.input_layernorm)
        self.k = copy.deepcopy(attn.k_proj)
        self.v = copy.deepcopy(attn.v_proj)
        self.k_norm = copy.deepcopy(attn.k_norm)

    def forward(self, x: torch.Tensor):
        h = self.norm(x)
        shape = (*h.shape[:-1], -1, self.hd)
        k = self.k_norm(self.k(h).view(shape)).transpose(1, 2)
        v = self.v(h).view(shape).transpose(1, 2)
        return k, v


def make_adapters(model, start: int) -> nn.ModuleDict:
    layers = model.model.layers
    return nn.ModuleDict({str(j): KVAdapter(layers[j]) for j in range(start, len(layers))})


class Manual:
    def __init__(self, model, yes_id: int, no_id: int) -> None:
        self.model = model
        self.base = model.model
        cfg = model.config
        self.L = cfg.num_hidden_layers
        self.d = cfg.hidden_size
        self.hq = cfg.num_attention_heads
        self.hkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.mi = cfg.intermediate_size
        self.vocab = cfg.vocab_size
        weight = model.lm_head.weight
        self.w = (weight[yes_id].float() - weight[no_id].float()).contiguous()
        self.device = weight.device
        self.q_macs = self.d * self.hq * self.hd
        self.kv_macs = self.d * 2 * self.hkv * self.hd
        self.o_macs = self.hq * self.hd * self.d
        self.mlp_macs = 3 * self.d * self.mi
        self.full_macs = self.q_macs + self.kv_macs + self.o_macs + self.mlp_macs
        self._adapters: dict[str, nn.ModuleDict] = {}

    def adapters_for(self, plan: Plan):
        if not plan.adapter:
            return None
        if plan.adapter not in self._adapters:
            state = torch.load(plan.adapter, map_location=self.device)
            mods = make_adapters(self.model, int(state["start"])).to(self.device)
            mods.load_state_dict(state["state"])
            self._adapters[plan.adapter] = mods.eval()
        return self._adapters[plan.adapter]

    def token_macs(self, plan: Plan, layer: int) -> float:
        """Linear-layer MACs of one fully computed token in one layer under a plan's sparsity."""
        xk, ck, mk = plan.x_keep, plan.cats_keep, plan.mid_keep
        third = self.d * self.mi
        mlp = third * xk + third * xk * ck + third * ck * mk
        density = plan.mlp_density.get(layer, 1.0)
        mlp = mlp * density
        return (self.q_macs + self.kv_macs) * xk + self.o_macs + mlp

    def _proj(self, layer, h, plan: Plan):
        attn = layer.self_attn
        h = topk_mask(h, plan.x_keep)
        shape = (*h.shape[:-1], -1, self.hd)
        q = attn.q_norm(attn.q_proj(h).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(h).view(shape)).transpose(1, 2)
        v = attn.v_proj(h).view(shape).transpose(1, 2)
        return q, k, v

    def _mlp(self, layer, x, plan: Plan):
        mlp = layer.mlp
        h = layer.post_attention_layernorm(x)
        if plan.x_keep >= 1 and plan.cats_keep >= 1 and plan.mid_keep >= 1:
            return mlp(h)
        h = topk_mask(h, plan.x_keep)
        a = F.silu(mlp.gate_proj(h))
        a = topk_mask(a, plan.cats_keep)
        z = topk_mask(a * mlp.up_proj(h), plan.mid_keep)
        return mlp.down_proj(z)

    @torch.no_grad()
    def prefix(self, ids: list[int], plan: Plan) -> tuple[list, int]:
        """The shared query prefix runs once per query, so sparsity plans leave it dense."""
        plan = Plan(skip=plan.skip)
        x = self.base.embed_tokens(torch.tensor([ids], device=self.device))
        pos = torch.arange(len(ids), device=self.device)[None]
        cos, sin = self.base.rotary_emb(x, pos)
        cache = []
        for j, layer in enumerate(self.base.layers):
            q, k, v = self._proj(layer, layer.input_layernorm(x), plan)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            cache.append((k, v))
            if j in plan.skip:
                continue
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, scale=layer.self_attn.scaling)
            x = x + layer.self_attn.o_proj(a.transpose(1, 2).reshape(1, len(ids), -1))
            x = x + self._mlp(layer, x, plan)
        return cache, len(ids)

    @torch.no_grad()
    def score(self, cache, P: int, conts: list[list[int]], S: int, plan: Plan, fixed_len: int = 0, positions=None):
        margin, online, offline = self.forward_batch(
            cache, P, conts, S, plan, fixed_len, self.adapters_for(plan), positions
        )
        return margin.tolist(), online.tolist(), offline.tolist()

    @torch.no_grad()
    def collect(self, cache, P: int, conts: list[list[int]], S: int, plan: Plan, fixed_len: int = 0):
        """Residuals after each layer in plan.collect: (B, n, d) last token, or (B, n, S+1, d) suffix tokens + doc mean."""
        self._collected = {}
        margin, online, offline = self.forward_batch(cache, P, conts, S, plan, fixed_len, self.adapters_for(plan))
        hidden = torch.stack([self._collected[j] for j in plan.collect], 1).half().cpu()
        return margin.tolist(), online.tolist(), offline.tolist(), hidden

    def forward_batch(self, cache, P: int, conts: list[list[int]], S: int, plan: Plan, fixed_len: int = 0, adapters=None, positions=None):
        B = len(conts)
        W = max(len(c) for c in conts)
        dev = self.device
        ids = torch.zeros((B, W), dtype=torch.long, device=dev)
        lens = torch.tensor([len(c) for c in conts], device=dev)
        for i, c in enumerate(conts):
            ids[i, : len(c)] = torch.tensor(c, device=dev)
        idx = torch.arange(W, device=dev)[None].expand(B, W)
        valid = idx < lens[:, None]
        is_suffix = valid & (idx >= (lens[:, None] - S))
        is_doc = valid & ~is_suffix
        if positions is None:
            t = idx
        else:
            t = torch.zeros((B, W), dtype=torch.long, device=dev)
            for i, p in enumerate(positions):
                t[i, : len(p)] = torch.tensor(p, device=dev)
        last = lens - 1
        rows = torch.arange(B, device=dev)

        x = self.base.embed_tokens(ids)
        cos, sin = self.base.rotary_emb(x, P + t)
        if plan.blind_layer is not None and plan.blind_shift:
            cos_off, sin_off = self.base.rotary_emb(x, fixed_len + t)
        alive = valid.clone()
        active = valid.clone()
        causal = torch.ones((W, W), dtype=torch.bool, device=dev).tril()
        pre_all = torch.ones((B, W, P), dtype=torch.bool, device=dev)
        flex_block, flex_key = (None, None)
        if plan.blind_layer is None and plan.prune_layer is None:
            flex_block, flex_key = flex_doc_block(P, lens, W)
        pre_blind = pre_all.clone()
        pre_blind[:, :, fixed_len:] = ~is_doc[:, :, None].expand(B, W, P - fixed_len)
        online = torch.zeros(B, dtype=torch.float64, device=dev)
        offline = torch.zeros(B, dtype=torch.float64, device=dev)
        attn_unit = 2 * self.hq * self.hd

        for j, layer in enumerate(self.base.layers):
            if j in plan.skip:
                continue
            blind = plan.blind_layer is not None and j < plan.blind_layer
            if plan.doc_stop is not None and j == plan.doc_stop:
                active = active & ~is_doc
            kv_only = alive & ~active
            per_tok = self.token_macs(plan, j)
            keymask = torch.cat([torch.ones((B, P), dtype=torch.bool, device=dev), alive], 1)
            if blind:
                pre = pre_blind
                cont_keys = causal[None] & alive[:, None, :]
                if not plan.blind_suffix_doc:
                    cont_keys = cont_keys & ~(is_suffix[:, :, None] & is_doc[:, None, :])
                mask = torch.cat([pre, cont_keys], 2)
                seen = mask.sum(2).double()
                doc_act = active & is_doc
                rest = active & ~is_doc
                offline += doc_act.sum(1).double() * per_tok + (seen * doc_act).sum(1) * attn_unit
                online += rest.sum(1).double() * per_tok + (seen * rest).sum(1) * attn_unit
                offline += kv_only.sum(1).double() * self.kv_macs
            else:
                mask = torch.cat([pre_all, causal[None].expand(B, W, W)], 2) & keymask[:, None, :]
                seen = (P + torch.cumsum(alive.long(), 1)).double()
                online += active.sum(1).double() * per_tok + (seen * active).sum(1) * attn_unit
                online += kv_only.sum(1).double() * self.kv_macs

            q0, k0, v = self._proj(layer, layer.input_layernorm(x), plan)
            if adapters is not None and str(j) in adapters:
                ka, va = adapters[str(j)](x)
                sel = kv_only[:, None, :, None]
                k0 = torch.where(sel, ka, k0)
                v = torch.where(sel, va, v)
            q, k = apply_rotary_pos_emb(q0, k0, cos, sin)
            pk, pv = cache[j]
            kf = torch.cat([pk.expand(B, -1, -1, -1), k], 2)
            vf = torch.cat([pv.expand(B, -1, -1, -1), v], 2)
            if flex_block is not None and not blind:
                a = flex_doc_attend(q, kf, vf, flex_block, flex_key, layer.self_attn.scaling, mask[:, None])
            else:
                a = F.scaled_dot_product_attention(q, kf, vf, attn_mask=mask[:, None], enable_gqa=True, scale=layer.self_attn.scaling)
            if blind and plan.blind_shift:
                qo, ko = apply_rotary_pos_emb(q0, k0, cos_off, sin_off)
                kfo = torch.cat([pk[:, :, :fixed_len].expand(B, -1, -1, -1), ko], 2)
                vfo = torch.cat([pv[:, :, :fixed_len].expand(B, -1, -1, -1), v], 2)
                mo = torch.cat([torch.ones((B, W, fixed_len), dtype=torch.bool, device=dev), causal[None] & (alive & is_doc)[:, None, :]], 2)
                ao = F.scaled_dot_product_attention(qo, kfo, vfo, attn_mask=mo[:, None], enable_gqa=True, scale=layer.self_attn.scaling)
                a = torch.where(is_doc[:, None, :, None], ao, a)
            x2 = x + layer.self_attn.o_proj(a.transpose(1, 2).reshape(B, W, -1))
            x2 = x2 + self._mlp(layer, x2, plan)
            x = torch.where(active[..., None], x2, x)
            if j in plan.collect:
                if plan.collect_pool:
                    suf = x[rows[:, None], lens[:, None] - S + torch.arange(S, device=dev)].float()
                    xn = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
                    dm = (xn * is_doc[..., None]).sum(1) / is_doc.sum(1).clamp(min=1)[:, None]
                    self._collected[j] = torch.cat([suf, dm[:, None]], 1)
                else:
                    self._collected[j] = x[rows, last].float()

            if plan.prune_layer is not None and j == plan.prune_layer:
                rep = self.hq // self.hkv
                kr = kf.repeat_interleave(rep, 1).float()
                if plan.prune_rows == "last":
                    qr = q[rows, :, last].float()[:, :, None]
                    sc = (qr @ kr.transpose(-1, -2)) * layer.self_attn.scaling
                    sc = sc.masked_fill(~keymask[:, None, None, :], float("-inf"))
                    imp = sc.softmax(-1)[:, :, 0].mean(1)[:, P:]
                else:
                    sc = (q.float() @ kr.transpose(-1, -2)) * layer.self_attn.scaling
                    sc = sc.masked_fill(~mask[:, None], float("-inf"))
                    pr = sc.softmax(-1).mean(1)[:, :, P:]
                    wt = is_suffix.float()
                    imp = (pr * wt[:, :, None]).sum(1) / wt.sum(1, keepdim=True)
                imp = imp.masked_fill(~is_doc, float("-inf"))
                ndoc = is_doc.sum(1)
                k_keep = torch.clamp(torch.ceil(ndoc.double() * plan.keep).long(), min=plan.keep_min)
                k_keep = torch.minimum(k_keep, ndoc)
                order = imp.argsort(1, descending=True)
                rank = torch.empty_like(order)
                rank.scatter_(1, order, torch.arange(W, device=dev)[None].expand(B, W))
                kept = rank < k_keep[:, None]
                drop = is_doc & ~kept
                alive = alive & ~drop
                active = active & ~drop
                online += attn_unit / 2 * keymask.sum(1).double()

        hl = self.base.norm(x[rows, last])
        margin = hl.float() @ self.w
        online += 2 * self.d
        return margin, online, offline

    def ckpt_margin(self, cache, P: int, conts: list[list[int]], S: int):
        """Default-plan margin with each layer recomputed on the backward pass."""
        from torch.utils.checkpoint import checkpoint

        B = len(conts)
        W = max(len(c) for c in conts)
        dev = self.device
        ids = torch.zeros((B, W), dtype=torch.long, device=dev)
        lens = torch.tensor([len(c) for c in conts], device=dev)
        for i, c in enumerate(conts):
            ids[i, : len(c)] = torch.tensor(c, device=dev)
        idx = torch.arange(W, device=dev)[None].expand(B, W)
        valid = idx < lens[:, None]
        x = self.base.embed_tokens(ids)
        cos, sin = self.base.rotary_emb(x, P + idx)
        causal = torch.ones((W, W), dtype=torch.bool, device=dev).tril()
        pre_all = torch.ones((B, W, P), dtype=torch.bool, device=dev)
        keymask = torch.cat([torch.ones((B, P), dtype=torch.bool, device=dev), valid], 1)
        mask = torch.cat([pre_all, causal[None].expand(B, W, W)], 2) & keymask[:, None, :]
        plan = Plan()

        def make_step(j):
            def step(x, cos, sin, pk, pv, mask, valid):
                layer = self.base.layers[j]
                q, k, v = self._proj(layer, layer.input_layernorm(x), plan)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                kf = torch.cat([pk.expand(x.shape[0], -1, -1, -1), k], 2)
                vf = torch.cat([pv.expand(x.shape[0], -1, -1, -1), v], 2)
                a = F.scaled_dot_product_attention(
                    q, kf, vf, attn_mask=mask[:, None], enable_gqa=True, scale=layer.self_attn.scaling
                )
                x2 = x + layer.self_attn.o_proj(a.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1))
                x2 = x2 + self._mlp(layer, x2, plan)
                return torch.where(valid[..., None], x2, x)
            return step

        for j in range(self.L):
            pk, pv = cache[j]
            x = checkpoint(make_step(j), x, cos, sin, pk, pv, mask, valid, use_reentrant=False)
        rows = torch.arange(B, device=dev)
        return self.base.norm(x[rows, lens - 1]).float() @ self.w

    def dense_macs(self, P: int, n: int) -> float:
        """MACs of the unchanged model on one continuation of n tokens after a cached prefix of P."""
        attn = 2 * self.hq * self.hd * (n * P + n * (n + 1) // 2)
        return self.L * (n * self.full_macs) + self.L * attn + 2 * self.d

    def hf_vocab_macs(self, n: int) -> float:
        """Extra MACs of the stock forward that projects every position onto the full vocabulary."""
        return float(n) * self.vocab * self.d


def score_query(engine: Manual, row: dict, plan: Plan, token_budget: int = 12000, hidden_out: list | None = None, positions=None):
    cache, P = engine.prefix(row["prefix"], plan)
    conts = row["conts"]
    S = row["suffix_len"]
    fixed = row.get("fixed_len", 0)
    order = sorted(range(len(conts)), key=lambda i: len(conts[i]))
    margins = [0.0] * len(conts)
    online = [0.0] * len(conts)
    offline = [0.0] * len(conts)
    start = 0
    while start < len(order):
        end = start
        while end < len(order) and (end - start + 1) * len(conts[order[end]]) <= token_budget:
            end += 1
        end = max(end, start + 1)
        idx = order[start:end]
        pos = None if positions is None else [positions[i] for i in idx]
        if plan.collect:
            m, on, off, hid = engine.collect(cache, P, [conts[i] for i in idx], S, plan, fixed)
            if hidden_out is not None:
                hidden_out.append((idx, hid))
        else:
            m, on, off = engine.score(cache, P, [conts[i] for i in idx], S, plan, fixed, pos)
        for i, mi, a, b in zip(idx, m, on, off):
            margins[i] = mi
            online[i] = a
            offline[i] = b
        start = end
    return margins, online, offline, P
