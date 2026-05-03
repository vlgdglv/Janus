"""
Speculative Jacobi Decoding (SJD) and Grouped Speculative Decoding (GSD) for
Janus image generation.

SJD   — classic token-level accept/reject using min(1, p_adv / p_draft).
GSD   — cluster-based accept/reject: the draft token is accepted when a cluster
         of visually-similar, logit-adjacent tokens has sufficient probability
         mass in the advanced distribution (Grouped Speculative Decoding, §3 of
         https://arxiv.org/pdf/2508.07747).

Both functions return (generated_tokens: LongTensor[image_token_num], stats).
"""

import copy
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from janus.models.modeling_vlm import MultiModalityCausalLM

IMG_VOCAB_SIZE = 16384

# GSD hyper-parameters (match the reference implementation)
GSD_G     = 14    # cluster half-width in logit-sorted order
GSD_P_THR = 0.15  # max |adv_prob_cluster - adv_prob_draft| to include a token
GSD_D_THR = 0.5   # max squared-L2 distance (normalized embeds) to include a token


# ── KV-cache helpers ──────────────────────────────────────────────────────────

def _kv_len(past: DynamicCache) -> int:
    return past.key_cache[0].shape[-2]


def _rollback_kv(past: DynamicCache, n: int) -> None:
    if n <= 0:
        return
    for i in range(len(past.key_cache)):
        past.key_cache[i] = past.key_cache[i][..., :-n, :]
        past.value_cache[i] = past.value_cache[i][..., :-n, :]


# ── CFG / embedding helpers ───────────────────────────────────────────────────

def _cfg_merge(raw: torch.Tensor, cfg_weight: float) -> torch.Tensor:
    """raw [2, *, V] → [*, V].  Batch row 0 = cond, row 1 = uncond."""
    return raw[1] + cfg_weight * (raw[0] - raw[1])


def _embed_cfg(mmgpt: MultiModalityCausalLM, tokens: torch.Tensor) -> torch.Tensor:
    """tokens [W] → [2, W, D]  (duplicated for the CFG cond/uncond batch)."""
    emb = mmgpt.prepare_gen_img_embeds(tokens)
    return torch.stack([emb, emb], dim=0)


def _pos_ids(base: int, W: int, device) -> torch.Tensor:
    return (base + torch.arange(W, device=device)).unsqueeze(0).expand(2, -1)


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class DecodeStats:
    total_tokens: int = 0
    total_fwd:    int = 0
    n_accepted:   int = 0
    n_rejected:   int = 0

    def report(self) -> dict:
        denom = max(1, self.n_accepted + self.n_rejected)
        return dict(
            total_tokens   = self.total_tokens,
            total_fwd      = self.total_fwd,
            tokens_per_fwd = round(self.total_tokens / max(1, self.total_fwd), 3),
            accept_rate    = round(self.n_accepted / denom, 3),
        )


# ── Text prefill ──────────────────────────────────────────────────────────────

def _text_prefill(
    mmgpt, vl_chat_processor, prompt: str,
    cfg_weight: float, temperature: float, device, stats: DecodeStats,
):
    """
    Embed + forward the text prompt.

    Returns
    -------
    past           : DynamicCache
    anchor_logits  : Tensor [V]   (raw CFG-merged logits for the 1st image token)
    anchor_probs   : Tensor [V]   (softmax'd, temperature-scaled)
    L_text         : int
    """
    tok  = vl_chat_processor.tokenizer
    ids  = torch.LongTensor(tok.encode(prompt)).to(device)
    toks = torch.stack([ids, ids], dim=0)
    toks[1, 1:-1] = vl_chat_processor.pad_id

    inp = mmgpt.language_model.get_input_embeddings()(toks)
    out = mmgpt.language_model.model(
        inputs_embeds=inp, use_cache=True, past_key_values=None
    )
    past   = out.past_key_values
    L_text = _kv_len(past)
    stats.total_fwd += 1

    raw            = mmgpt.gen_head(out.last_hidden_state[:, -1, :])  # [2, V]
    anchor_logits  = _cfg_merge(raw, cfg_weight)                       # [V]
    anchor_probs   = F.softmax(anchor_logits / temperature, dim=-1)   # [V]

    return past, anchor_logits, anchor_probs, L_text


# ── Rejection resampling ──────────────────────────────────────────────────────

def _resample_rejected(
    adv_prob:   torch.Tensor,   # [V]
    draft_prob: torch.Tensor,   # [V]
    generator:  torch.Generator | None = None,
) -> int:
    diff = (adv_prob - draft_prob).clamp(min=0.0)
    s    = diff.sum()
    src  = diff / s if s > 0 else adv_prob
    return int(torch.multinomial(src, 1, generator=generator))


# ── SJD acceptance ────────────────────────────────────────────────────────────

def _sjd_accept_p(
    anchor_logits: torch.Tensor,  # [V]  (unused, kept for uniform interface)
    anchor_probs:  torch.Tensor,  # [V]
    p_d:           torch.Tensor,  # [V]
    cls_idx:       int,
    **_kw,
) -> float:
    return float((anchor_probs[cls_idx] / p_d[cls_idx]).clamp(max=1.0))


# ── GSD acceptance ────────────────────────────────────────────────────────────

def _gsd_accept_p(
    anchor_logits: torch.Tensor,   # [V] raw CFG-merged logits
    anchor_probs:  torch.Tensor,   # [V] temperature-scaled softmax
    p_d:           torch.Tensor,   # [V] draft distribution
    cls_idx:       int,
    img_emb_norm:  torch.Tensor,   # [V, D] L2-normalised codebook embeddings
    G:    int   = GSD_G,
    p_thr: float = GSD_P_THR,
    d_thr: float = GSD_D_THR,
) -> float:
    """
    GSD cluster-based acceptance probability.

    Cluster C_idx = the G tokens centred on the draft token's rank in
    logit-sorted order.  Tokens in C_idx are further filtered by:
      • probability similarity  |adv_prob[c] - adv_prob[cls_idx]| < p_thr
      • embedding distance       ||emb_norm[cls_idx] - emb_norm[c]||² < d_thr
    """
    V = anchor_logits.shape[-1]

    # Rank of cls_idx in ascending-logit order.
    logit_sort = torch.argsort(anchor_logits)          # [V] ascending
    rank       = (logit_sort == cls_idx).nonzero(as_tuple=True)[0]
    if rank.numel() == 0:
        return float((anchor_probs[cls_idx] / p_d[cls_idx]).clamp(max=1.0))
    cur = int(rank[0])

    lo  = max(0, cur - G // 2)
    hi  = min(V, cur + G // 2)
    cluster = logit_sort[lo:hi]                        # [≤G]

    # ── Probability filter ────────────────────────────────────────────────────
    is_in_p = ((anchor_probs[cluster] - anchor_probs[cls_idx]).abs() < p_thr).float()

    # ── Embedding-distance filter ─────────────────────────────────────────────
    d_emb    = img_emb_norm[cls_idx]            # [D]
    c_embs   = img_emb_norm[cluster]            # [≤G, D]
    dists    = ((d_emb - c_embs) ** 2).sum(-1)  # [≤G] squared L2 in unit sphere
    is_in_d  = (dists < d_thr).float()

    mask = is_in_p * is_in_d

    adv_super   = (anchor_probs[cluster] * mask).sum()
    draft_super = (p_d[cluster] * mask).sum()

    if float(draft_super) <= 0:
        # Degenerate cluster: fall back to single-token SJD
        return float((anchor_probs[cls_idx] / p_d[cls_idx]).clamp(max=1.0))

    return float((adv_super / draft_super).clamp(max=1.0))


# ══════════════════════════════════════════════════════════════════════════════
#  Shared Jacobi window
# ══════════════════════════════════════════════════════════════════════════════

def _jacobi_window(
    mmgpt:         MultiModalityCausalLM,
    past:          DynamicCache,
    anchor_logits: torch.Tensor,   # [V]  raw CFG-merged logits for the anchor
    anchor_probs:  torch.Tensor,   # [V]  temperature-scaled probs
    abs_pos:       int,
    W:             int,
    temperature:   float,
    cfg_weight:    float,
    max_iter:      int,
    device,
    generator:     torch.Generator | None,
    stats:         DecodeStats,
    accept_fn:     Callable,       # (anchor_logits, anchor_probs, p_d, cls_idx, **kw) → float
    accept_kw:     dict,           # extra kwargs forwarded to accept_fn
):
    """
    Fill *W* image tokens at *abs_pos* with Jacobi iteration + accept/reject.

    The KV cache *past* is updated in-place (W new positions appended on exit).

    Returns
    -------
    accepted       : list[int]    length W
    past           : DynamicCache
    next_al        : Tensor [V]   raw logits for position abs_pos + W
    next_ap        : Tensor [V]   probs  for position abs_pos + W
    """
    V        = IMG_VOCAB_SIZE
    accepted : list[int] = []

    sub_pos  = abs_pos
    sub_al   = anchor_logits    # [V]  anchor raw logits
    sub_ap   = anchor_probs     # [V]  anchor probs

    # ── initial draft ─────────────────────────────────────────────────────────
    # All positions sampled from the anchor distribution.
    # Uniform drafts (1/V) have ~50% rejection on image tokens, meaning ~every
    # window needs a correction forward; anchor drafts are much better priors
    # for spatially-correlated patches and recover most of the speedup.
    draft       = torch.multinomial(sub_ap, W, replacement=True, generator=generator)
    draft_probs = sub_ap.unsqueeze(0).expand(W, -1).clone()

    for _it in range(max_iter):
        R               = W - len(accepted)
        sub_draft       = draft[:R]
        sub_draft_probs = draft_probs[:R]

        # ── forward pass ─────────────────────────────────────────────────────
        step_emb = _embed_cfg(mmgpt, sub_draft)
        pos_ids  = _pos_ids(sub_pos, R, device)

        out = mmgpt.language_model.model(
            inputs_embeds=step_emb,
            past_key_values=past,
            position_ids=pos_ids,
            use_cache=True,
        )
        past = out.past_key_values
        stats.total_fwd += 1

        raw       = mmgpt.gen_head(out.last_hidden_state)         # [2, R, V]
        adv_logit = _cfg_merge(raw, cfg_weight)                    # [R, V]
        adv_probs = F.softmax(adv_logit / temperature, dim=-1)    # [R, V]

        # ── accept / reject ───────────────────────────────────────────────────
        # Anchor for position j within the sub-window:
        #   j == 0  →  (sub_al, sub_ap)       – before seeing any draft token
        #   j  > 0  →  (adv_logit[j-1], adv_probs[j-1])

        j_reject  = R          # first position that was rejected (R = none)
        resampled = None

        for j in range(R):
            al  = sub_al          if j == 0 else adv_logit[j - 1]
            ap  = sub_ap          if j == 0 else adv_probs[j - 1]
            p_d = sub_draft_probs[j]
            d_j = int(sub_draft[j])

            p_accept = accept_fn(al, ap, p_d, d_j, **accept_kw)
            r_val    = float(torch.rand(1, device=device, generator=generator))

            if r_val < p_accept:
                stats.n_accepted += 1
            else:
                resampled = _resample_rejected(ap, p_d, generator)
                j_reject  = j
                stats.n_rejected += 1
                break

        # Commit accepted draft tokens; roll back trailing positions not committed.
        for j in range(j_reject):
            accepted.append(int(sub_draft[j]))
        # Keep j_reject's KV entry (d_j approximation); remove j_reject+1..R-1.
        n_keep = j_reject + 1 if resampled is not None else R
        _rollback_kv(past, R - n_keep)

        if resampled is not None:
            # Keep draft k/v at position j_reject as an approximation; the anchor
            # draft ensures d_j is a plausible token (≈ sub_ap), so the KV error
            # is small. Uniform drafts produced strips/collapse because rejected
            # tokens were arbitrary (any of 16384); anchor drafts bound that error.
            accepted.append(resampled)
            # Anchor for next position comes from the forward we already ran.
            # adv_logit[j_reject] = p(t | ..., d_0..d_{j_reject}); since d_{j_reject}
            # ≈ r_{j_reject} in distribution, this is a good approximation.
            sub_al  = adv_logit[j_reject].clone()
            sub_ap  = F.softmax(sub_al / temperature, dim=-1)
            sub_pos += j_reject + 1
        else:
            sub_al  = adv_logit[-1].clone()
            sub_ap  = adv_probs[-1].clone()
            sub_pos += R

        if len(accepted) == W:
            return accepted, past, sub_al, sub_ap

        # ── next iteration ────────────────────────────────────────────────────
        # sub_al / sub_ap are now correct for position sub_pos.
        # Draft all remaining positions from the (corrected) anchor.
        new_R          = W - len(accepted)
        new_draft      = torch.multinomial(sub_ap, new_R, replacement=True, generator=generator)
        new_draft_prob = sub_ap.unsqueeze(0).expand(new_R, -1).clone()
        draft       = new_draft
        draft_probs = new_draft_prob

    # max_iter exceeded: AR fallback for stragglers
    remaining = W - len(accepted)
    for _ in range(remaining):
        t       = int(torch.multinomial(sub_ap, 1, generator=generator))
        tok_emb = _embed_cfg(mmgpt, torch.tensor([t], dtype=torch.long, device=device))
        out     = mmgpt.language_model.model(
            inputs_embeds=tok_emb, past_key_values=past,
            position_ids=_pos_ids(sub_pos, 1, device), use_cache=True,
        )
        past    = out.past_key_values
        stats.total_fwd += 1
        raw     = mmgpt.gen_head(out.last_hidden_state[:, -1, :])
        sub_al  = _cfg_merge(raw, cfg_weight)
        sub_ap  = F.softmax(sub_al / temperature, dim=-1)
        accepted.append(t)
        sub_pos += 1

    return accepted, past, sub_al, sub_ap


# ══════════════════════════════════════════════════════════════════════════════
#  Public API – SJD
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def generate_sjd(
    mmgpt:               MultiModalityCausalLM,
    vl_chat_processor,
    prompt:              str,
    temperature:         float = 1.0,
    cfg_weight:          float = 5.0,
    image_token_num:     int   = 576,
    img_size:            int   = 384,
    patch_size:          int   = 16,
    jacobi_window:       int   = 16,
    max_iter_per_window: int   = 20,
    seed:                int | None = None,
) -> tuple[torch.Tensor, DecodeStats]:
    """
    Generate one image with Speculative Jacobi Decoding.
    Accept criterion: min(1, p_adv(draft_token) / p_draft(draft_token)).
    """
    device    = next(mmgpt.language_model.parameters()).device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    stats = DecodeStats()
    past, al, ap, L_text = _text_prefill(
        mmgpt, vl_chat_processor, prompt, cfg_weight, temperature, device, stats
    )

    generated = torch.zeros(image_token_num, dtype=torch.long, device=device)
    cur_pos   = 0

    while cur_pos < image_token_num:
        W = min(jacobi_window, image_token_num - cur_pos)

        accepted, past, al, ap = _jacobi_window(
            mmgpt=mmgpt, past=past,
            anchor_logits=al, anchor_probs=ap,
            abs_pos=L_text + cur_pos,
            W=W, temperature=temperature, cfg_weight=cfg_weight,
            max_iter=max_iter_per_window,
            device=device, generator=generator, stats=stats,
            accept_fn=_sjd_accept_p, accept_kw={},
        )

        for k, t in enumerate(accepted):
            generated[cur_pos + k] = t
        stats.total_tokens += W
        cur_pos            += W

    return generated, stats


# ══════════════════════════════════════════════════════════════════════════════
#  Public API – GSD
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def generate_gsd(
    mmgpt:               MultiModalityCausalLM,
    vl_chat_processor,
    prompt:              str,
    temperature:         float = 1.0,
    cfg_weight:          float = 5.0,
    image_token_num:     int   = 576,
    img_size:            int   = 384,
    patch_size:          int   = 16,
    jacobi_window:       int   = 16,
    max_iter_per_window: int   = 20,
    G:                   int   = GSD_G,
    p_thr:               float = GSD_P_THR,
    d_thr:               float = GSD_D_THR,
    seed:                int | None = None,
) -> tuple[torch.Tensor, DecodeStats]:
    """
    Generate one image with Grouped Speculative Decoding.

    The draft token is accepted if the *cluster* of G logit-adjacent tokens that
    surround it has sufficient probability mass in the advanced distribution,
    after filtering by probability similarity and visual-embedding distance.
    """
    device    = next(mmgpt.language_model.parameters()).device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    stats = DecodeStats()

    # ── precompute normalised codebook embeddings ─────────────────────────────
    raw_emb      = mmgpt.gen_embed.weight.detach().float()     # [V, D]
    norms        = raw_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    img_emb_norm = (raw_emb / norms).to(device)                # [V, D] unit vectors

    accept_kw = dict(img_emb_norm=img_emb_norm, G=G, p_thr=p_thr, d_thr=d_thr)

    past, al, ap, L_text = _text_prefill(
        mmgpt, vl_chat_processor, prompt, cfg_weight, temperature, device, stats
    )

    generated = torch.zeros(image_token_num, dtype=torch.long, device=device)
    cur_pos   = 0

    while cur_pos < image_token_num:
        W = min(jacobi_window, image_token_num - cur_pos)

        accepted, past, al, ap = _jacobi_window(
            mmgpt=mmgpt, past=past,
            anchor_logits=al, anchor_probs=ap,
            abs_pos=L_text + cur_pos,
            W=W, temperature=temperature, cfg_weight=cfg_weight,
            max_iter=max_iter_per_window,
            device=device, generator=generator, stats=stats,
            accept_fn=_gsd_accept_p, accept_kw=accept_kw,
        )

        for k, t in enumerate(accepted):
            generated[cur_pos + k] = t
        stats.total_tokens += W
        cur_pos            += W

    return generated, stats


# ══════════════════════════════════════════════════════════════════════════════
#  Decode helpers
# ══════════════════════════════════════════════════════════════════════════════

def tokens_to_image(
    mmgpt:      MultiModalityCausalLM,
    tokens:     torch.Tensor,
    img_size:   int = 384,
    patch_size: int = 16,
) -> np.ndarray:
    H = W = img_size // patch_size
    dec = mmgpt.gen_vision_model.decode_code(
        tokens.unsqueeze(0).to(dtype=torch.int), shape=[1, 8, H, W],
    )
    dec = dec.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    return np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)[0]


def save_image(arr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    PIL.Image.fromarray(arr).save(path)
