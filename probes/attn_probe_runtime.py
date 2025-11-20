# attn_probe_runtime.py
import contextlib, contextvars, math, torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class TokenStat:
    step: int
    pos_1d: int
    row: Optional[int]
    col: Optional[int]
    entropy: float
    top1_p: float
    topk_mass: float  # e.g., k=5
    layer_entropies: Optional[List[float]] = None  # for E
    locality_r_by_layer_head: Optional[Dict[Tuple[int,int], float]] = None  # for B
    attn_sparsity_by_layer_head: Optional[Dict[Tuple[int,int], float]] = None  # for C


_current_layer = contextvars.ContextVar("llama_layer_idx", default=-1)
_probe_ctx = contextvars.ContextVar("probe_ctx", default=None)

def _default_probe_callback(*, layer_idx, sdpa_probs, probe_ctx):
    a = sdpa_probs[0, :, -1, :].detach()
    probe_ctx.setdefault("attn_by_layer", {})[layer_idx] = a

_user_callback = _default_probe_callback

def set_probe_callback(cb):
    global _user_callback
    _user_callback = cb

def attach_layer_indices(model):
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            setattr(layer.self_attn, "_layer_idx", i)

_orig_sdpa = F.scaled_dot_product_attention

def _sdpa_with_probe(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    out = _orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    ctx = _probe_ctx.get()
    if ctx is not None:
        with torch.no_grad():
            d = q.size(-1)
            s = (1.0 / math.sqrt(d)) if scale is None else scale
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * s
            if attn_mask is not None:
                scores = scores + attn_mask
            probs = torch.softmax(scores, dim=-1)  # [B, nH, Tq, Tk]
            _user_callback(layer_idx=_current_layer.get(), sdpa_probs=probs, probe_ctx=ctx)
    return out

def _wrap_attn_forward(attn_cls):
    _orig = attn_cls.forward
    def _wrapped(self, *args, **kwargs):
        token = _current_layer.set(getattr(self, "_layer_idx", -1))
        try:
            return _orig(self, *args, **kwargs)
        finally:
            _current_layer.reset(token)
    return _wrapped

@contextlib.contextmanager
def enable_attention_probe(model, *, ctx_payload:dict):
    """
    ctx_payload: 任意你想在回调里用到的信息，比如：
      {"step":int, "pos_1d":int, "img_start":int, "H":int, "W":int}
    """
    from transformers.models.llama import modeling_llama as ml
    LlamaAttention = ml.LlamaAttention
    orig_attn_fwd = LlamaAttention.forward
    LlamaAttention.forward = _wrap_attn_forward(LlamaAttention)

    
    if hasattr(model.config, "attn_implementation"):
        old_impl = model.config.attn_implementation
        # model.config.attn_implementation = "sdpa"
    else:
        old_impl = None

    sdpa_bk = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _sdpa_with_probe

    token_ctx = _probe_ctx.set(ctx_payload)
    try:
        yield
    finally:
        _probe_ctx.reset(token_ctx)
        F.scaled_dot_product_attention = sdpa_bk
        LlamaAttention.forward = orig_attn_fwd
        if old_impl is not None:
            model.config.attn_implementation = old_impl
