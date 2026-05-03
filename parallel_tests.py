# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os, time
import PIL.Image

# for probe
from probes.attn_probe_runtime import attach_layer_indices, enable_attention_probe, set_probe_callback, TokenStat
# from probes.layer_hooks import register_lasttoken_hooks
import torch
import torch.nn.functional as F
from datetime import datetime

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# specify the path to the model
# model_path = "deepseek-ai/Janus-1.3B"
model_path = "/home/vlgd/Models/Janus-Pro-1B/"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


target_size = 384
patch_size = 16  
image_token_num_per_image = 576
H, W = 24, 24

q_image_content_conditions = [
    # "A black Honda motorcycle parked in front of a garage.",
    'A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.',
    'close-up two birds on a tree branch, background of blue sky with cumulus clouds and rising sun, 4k, realistic',
    'Three penguins in yellow construction helmets, building a sandcastle on a tropical beach, one holding a blueprint, the ocean behind them glowing in soft blue hues under the setting sun, hyperrealistic textures, playful and cinematic',
    'Deep in the jungle where a rusty robot is abndoned , 4k ,realistic, photography',
    'animation art work, A cheese burger on the sky with birds, bright, detailed',
    'Apple castle on the grass, realistic, 4k, detailed, photography',
    'A mischievous hippo playing soccer, realistic, 4k, detailed, photography',
    'Truck full of vegetables, afternoon, 4k, photography, bright color,',
    'Masterpiece, 4k, photography, bright background, market selling fresh fruits',
    'photo, photography, realistic, very detailed, Amsterdam, center fancy sports car, afternoon, realistic. sharp, bright, film grain, high contrast',

    'dystopic civilization beautiful landscape, morning, woman, very intricate, very detailed, sharp, bright, colorful',
    'A single coffee on a dinner plate on a table, 4k, detailed, photography',
    'A cat in a lab coat, standing in front of a chalkboard full of complex equations, realistic, 4k',
    'Pixel art, A mushroom kingdom, glowing, masterpiece',
    'Japanese woman in a floral-pattern summer dress sitting on an old boat beached on a tropical island, overlooking a majestic azure blue ocean with gentle waves, landscape, sunset. Impressionistic',
    'a_skynet_cyberdyne_craft, the image is featuring a futuristic, highly advanced jet fighter drone flying rapidly at altitude thporugh stormclouds, silhouetted, chiascuro, sunset., realistic, 4k',

    'abstract oil painting, gradient vibrant neon colour, rough, textural, broad brush strokes, a sleek spaceship traversing interstellar space, detailed night sky with stars and nebulas',
    'photo, photography, Fujifilm XT-4 Viltrox, Budapest, Hungary landscape, sunset, very intricate, very detailed, realistic. sharp, bright, colorful, film grain, high contrast',
    'A stylized clay cartoon character, a small, adorable humanoid figure with a skull head, riding a miniature motorcycle., detailed',
    'animation art work, cute cat boxing with silly dog, bright',
    'Pumpkin carraige on the road, 4k, realistic, photography',
    'photography, photo of a war pilot walking to his war plane on sunset, taken from behind, 4k, realistic',

    'animation art work, huge sand castle made by dwarfs, 4k, realistic',
    '4k, realistic, photography, Giant Tree on the hill, afternoon',
]

def build_prompt(desc: str) -> str:
    conv = [{"role": "User", "content": desc}, {"role": "Assistant", "content": ""}]
    s = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conv,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return s + vl_chat_processor.image_start_tag

def slugify_first_words(text: str, n=6) -> str:
    import re
    words = text.strip().split()
    base = "-".join(words[:n]).lower()
    base = re.sub(r"[^a-z0-9\-]+", "", base)
    return base or "img"

@torch.inference_mode()
def baseline_gt_logits(mm, proc, prompt, H, W, cfg_weight=5.0, temperature=1.0, do_sample=True):
    tok = proc.tokenizer
    ids = torch.LongTensor(tok.encode(prompt)).to("cuda")
    toks = torch.stack([ids, ids]).to("cuda")
    toks[1, 1:-1] = proc.pad_id
    
    inp = mm.language_model.get_input_embeddings()(toks)
    T = H * W
    gt_tokens, gt_logits = [], []
    probs_all = []
    generated_tokens = torch.zeros((T), dtype=torch.int).cuda()
    
    g = torch.Generator(device="cuda")
    g.manual_seed(42)

    past = None
    for t in range(T):
        out = mm.language_model.model(inputs_embeds=inp, use_cache=True, past_key_values=past)
        past = out.past_key_values
        hs = out.last_hidden_state[:, -1, :]
        log_all = mm.gen_head(hs)
        
        log = log_all[1:2] + cfg_weight * (log_all[0:1] - log_all[1:2])
        probs = torch.softmax(log / temperature, dim=-1)
        probs_all.append(probs.squeeze().float().detach().cpu())
    
        if do_sample:
            nxt = torch.multinomial(probs, num_samples=1, generator=g)
        else:
            nxt = probs.argmax(dim=-1, keepdim=True)
            
        generated_tokens[t] = nxt
        gt_tokens.append(int(nxt))
        both = torch.cat([nxt, nxt], dim=0).view(-1)
        img_emb = mm.prepare_gen_img_embeds(both)
        inp = img_emb.unsqueeze(1)
      
    return generated_tokens, probs_all

def cfg_merge(log_all, cfg_weight=5.0):
    return log_all[1:2] + cfg_weight * (log_all[0:1] - log_all[1:2])

def _raster_idx(L0: int, W: int, r: int, c: int) -> int:
    return L0 + r * W + c

def _to_cache(past):
    import copy
    try:
        from transformers.cache_utils import Cache, DynamicCache
    except Exception:
        return tuple((k.detach().clone(), v.detach().clone()) for (k, v) in past) if past is not None else None
    if past is None:
        return None
    if isinstance(past, Cache):
        return copy.deepcopy(past)
    return DynamicCache.from_legacy_cache(
        tuple((k.detach().clone(), v.detach().clone()) for (k, v) in past)
    )

def _build_visible_indices_for_column(L0, W, i, c, policy="row_prefix", neighbor_k=3):
    vis = list(range(L0 + W * max(i-1, 1))) 
    if policy == "col_only":
        for rr in range(0, i):
            vis.append(_raster_idx(L0, W, rr, c))
    elif policy == "row_prefix":
        # for rr in range(0, i-1):
        #     vis.append(_raster_idx(L0, W, rr, c))
        for kk in range(0, W):
            vis.append(_raster_idx(L0, W, i-1, kk))
        for kk in range(0, c):
            vis.append(_raster_idx(L0, W, i, kk))
        # for kk in range(c, min(W, c+ neighbor_k + 1)):
        #     vis.append(_raster_idx(L0, W, i, kk))
    elif policy == "col_plus_neighbors":
        for rr in range(0, i):
            vis.append(_raster_idx(L0, W, rr, c))
        if neighbor_k > 0 and i-1 >= 0:
            for off in range(1, neighbor_k+1):
                k1 = max(0, c-off); k2 = min(W-1, c+off)
                vis.append(_raster_idx(L0, W, i-1, k1))
                vis.append(_raster_idx(L0, W, i-1, k2))
    else:
        raise ValueError(f"unknown policy={policy}")
    return sorted(set(vis))


@torch.inference_mode()
def baseline_row_parallel_logits(
    mm, proc, prompt, H, W,
    cfg_weight=5.0, temperature=1.0,
    ar_row=1, do_sample: bool = True,
    mask_policy: str = "row_prefix",
    neighbor_k: int = 0,
):
    device = next(mm.language_model.parameters()).device

    tok = proc.tokenizer
    ids = torch.LongTensor(tok.encode(prompt)).to("cuda")
    
    toks = torch.stack([ids, ids]).to("cuda")
    toks[1, 1:-1] = proc.pad_id

    inp = mm.language_model.get_input_embeddings()(toks)
    L_prefill = inp.shape[1]

    T = H * W
    gt_tokens, probs_all = [], []
    generated_tokens = torch.zeros((T), dtype=torch.int64, device="cuda")

    past = None
    pos_cur = L_prefill
    prev_row_embs = []

    g = torch.Generator(device="cuda")
    g.manual_seed(42)
    
    for c in range(W * ar_row):
        out = mm.language_model.model(
            inputs_embeds=inp,
            use_cache=True,
            past_key_values=past,
        )
        hs, past = out.last_hidden_state, out.past_key_values
        last_h = hs[:, -1, :]                           # [2, D]
        log = cfg_merge(mm.gen_head(last_h), cfg_weight).squeeze(0) # [V]
       
        probs = torch.softmax(log / temperature, dim=-1)
        probs_all.append(probs.squeeze().float().detach().cpu())

        if do_sample:
            nxt = torch.multinomial(probs, num_samples=1, generator=g).squeeze(0) # generator=g
        else:
            nxt = probs.argmax(dim=-1, keepdim=True)

        token_id = int(nxt)
        generated_tokens[0 * W + c] = token_id
        gt_tokens.append(token_id)

        both = torch.tensor([token_id, token_id], device="cuda").view(-1)
        img_emb = mm.prepare_gen_img_embeds(both)       # [2, D]
        prev_row_embs.append(img_emb[0].detach())       # [D]

        inp = img_emb.unsqueeze(1)                      # [2,1,D]
        pos_cur += 1

    prev_row_embs = prev_row_embs[-W:]
    prev_row_embs.insert(0, prev_row_embs.pop())

    for r in range(ar_row, H):
        # T_k = past[0][0].shape[2]
        pos_base = L_prefill + r * W
        
        # Just use default attention mask
        # attn_mask = torch.full((2, 1, W, K_total), torch.finfo(torch.float16).min, device=device) #
        # for c in range(W):
        #     vis = _build_visible_indices_for_column(L_prefill, W, r, c, mask_policy, neighbor_k)
        #     if len(vis):
        #         idx = torch.as_tensor(vis, device=device, dtype=torch.long)
        #         attn_mask[0, 0, c, idx] = 0.0
        #         attn_mask[1, 0, c, idx] = 0.0

        for _ in range(3):
            row_cond = torch.stack(prev_row_embs, dim=0).to(device)          # [W, D]
            step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous() # [2, W, D]
            pos_ids = (torch.arange(W, device=device, dtype=torch.long) + pos_base).unsqueeze(0).expand(2, -1)
            out_prop = mm.language_model.model(
                inputs_embeds=step_emb,          # [2, W, D]
                past_key_values=_to_cache(past),
                # attention_mask=attn_mask,        # [2, 1, W, T_k+W], 0/-inf
                position_ids=pos_ids,            # [2, W]
                use_cache=True
            )

            log_q = cfg_merge(mm.gen_head(out_prop.last_hidden_state), cfg_weight).squeeze(0)
            q_probs  = F.softmax(log_q / temperature, dim=-1)         # [W, V]
            proposal = torch.multinomial(q_probs, 1, generator=g).squeeze(-1)      # [W]
            prev_row_embs = list(mm.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0].unbind(dim=0))
            prev_row_embs.insert(0, prev_row_embs.pop())

        
        row_cond = torch.stack(prev_row_embs, dim=0).to(device)          # [W, D]
        step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous() # [2, W, D]
        pos_ids = (torch.arange(W, device=device, dtype=torch.long) + pos_base).unsqueeze(0).expand(2, -1)
        out_prop = mm.language_model.model(
            inputs_embeds=step_emb,
            past_key_values=past,
            position_ids=pos_ids,
            use_cache=True
        )
        log_q = cfg_merge(mm.gen_head(out_prop.last_hidden_state), cfg_weight).squeeze(0)
        # log_q = mm.gen_head(out_prop.last_hidden_state)
        # log_q = log_q[1:2] + cfg_weight * (log_q[0:1] - log_q[1:2])
        # log_q = log_q.squeeze(0)

        q_probs  = F.softmax(log_q / (temperature), dim=-1)   
        for prob in q_probs:
            probs_all.append(prob.float().detach().cpu())

        if do_sample:      # [W, V]
            proposal = torch.multinomial(q_probs, 1, generator=g).squeeze(-1)      # [W]
        else:
            proposal = q_probs.argmax(dim=-1)
        
        prev_row_embs = list(mm.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0].unbind(dim=0))

        final_row = proposal.clone()
    
        for c in range(W):
            generated_tokens[r * W + c] = int(final_row[c])

        prev_row_embs.insert(0, prev_row_embs.pop())

    return generated_tokens, probs_all


@torch.inference_mode()
def spec_row_parallel_logits(
    mm, proc, prompt, H, W,
    cfg_weight=5.0, temperature=1.0,
    row_parallel=True,
    mask_policy: str = "row_prefix",
    neighbor_k: int = 0,
    mask_is_bool: bool = True,
):
    device = next(mm.language_model.parameters()).device

    tok = proc.tokenizer
    ids = torch.LongTensor(tok.encode(prompt)).to("cuda")
    
    toks = torch.stack([ids, ids]).to("cuda")
    toks[1, 1:-1] = proc.pad_id

    inp = mm.language_model.get_input_embeddings()(toks)
    L_prefill = inp.shape[1]

    T = H * W
    gt_tokens, gt_logits = [], []
    generated_tokens = torch.zeros((T), dtype=torch.int64, device="cuda")

    past = None
    pos_cur = L_prefill
    prev_row_embs, prev_row_tokens = [], []
    
    x_row = 1

    for c in range(W * x_row):
        out = mm.language_model.model(
            inputs_embeds=inp,
            use_cache=True,
            past_key_values=past,
        )
        hs, past = out.last_hidden_state, out.past_key_values
        last_h = hs[:, -1, :]                           # [2, D]
        log = cfg_merge(mm.gen_head(last_h), cfg_weight).squeeze(0) # [V]
        
        gt_logits.append(log.detach().cpu())
        probs = F.softmax(log / temperature, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).squeeze(0) # generator=g
        token_id = int(nxt)
        generated_tokens[0 * W + c] = token_id
        prev_row_tokens.append(token_id)

        both = torch.tensor([token_id, token_id], device="cuda").view(-1)
        img_emb = mm.prepare_gen_img_embeds(both)       # [2, D]
        inp = img_emb.unsqueeze(1)                      # [2

    prev_row_tokens = prev_row_tokens[-W:]
    # prev_row_tokens.insert(0, prev_row_tokens.pop())
    # img_emb = mm.prepare_gen_img_embeds(torch.tensor([prev_row_tokens, prev_row_tokens], device="cuda"))       # [2, D]

    draft_input_ids = torch.tensor(prev_row_tokens, device=device)
    probe_ctx = {
        "accept_ratio": 0.0,
        "avg_accept_length": W * x_row,
        "forward_cnt": W * x_row,
        "verify_cnt": 0,
        "reject_cnt": 0
    }

    for r in range(x_row, H):
        probing = {
            "accept_ratio": 0.0,
            "accept_length": 0,
            "forward_cnt": 0,
        }

        start_cnt = L_prefill + r * W 
        start_pos = start_cnt # Index of the starting position for this row
        accepted_tokens = []
        prev_token_prob = None # [W, V]

        window_size = W
        while len(accepted_tokens) < W:
            input_embs_for_check = mm.prepare_gen_img_embeds(draft_input_ids.unsqueeze(0).expand(2, -1)) # [2, W, D]
            pos_ids = (torch.arange(window_size, device=device, dtype=torch.long) + start_pos).unsqueeze(0).expand(2, -1)
            # print(pos_ids)
            mm_out = mm.language_model.model(
                inputs_embeds=input_embs_for_check, # [2, W, D]
                past_key_values=past,
                position_ids=pos_ids,
                use_cache=True
            )
            probing["forward_cnt"] += 1

            hs = mm_out.last_hidden_state           # [2, W, D]
            past = mm_out.past_key_values
            log = cfg_merge(mm.gen_head(hs), cfg_weight).squeeze(0)
            curr_token_probs  = F.softmax(log / temperature, dim=-1)                # [W, V]
            curr_token_ids = torch.multinomial(curr_token_probs, 1).squeeze(-1)     # [W]

            if prev_token_prob is None:
                # First token is always accepted
                accepted_tokens.append(int(curr_token_ids[0]))
                prev_token_prob = curr_token_probs
                prev_token_ids = curr_token_ids
                draft_input_ids = curr_token_ids[:-1]
                window_size = W - len(accepted_tokens)
                start_pos = start_cnt + len(accepted_tokens)
                new_accepted_token = 1
            else:
                accepted_tokens.append(int(curr_token_ids[0]))
                new_accepted_token = 1
                if len(accepted_tokens) == W:
                    break
                prev_token_prob = prev_token_prob[1:]
                prev_token_ids = prev_token_ids[1:]
                verify_len = curr_token_ids.shape[0]

                random_prob = torch.rand(verify_len, device=device)
                first_unmatched_token = -1
                
                
                # prev_token = prev_token_ids[vi]
                # prev_prob = prev_token_prob[vi][prev_token]
                # curr_prob = curr_token_probs[vi][prev_token]
                # ex_prob = (curr_prob / prev_prob).clamp(max=1.0)
                
                for vi in range(1, verify_len):
                    probe_ctx["verify_cnt"] += 1
                    curr_token = curr_token_ids[vi]
                    prev_prob = prev_token_prob[vi][curr_token]
                    curr_prob = curr_token_probs[vi][curr_token]
                    ex_prob = prev_prob/curr_prob
                    
                    beta = 0.1
                    p_floor = 0.1
                    p_cap   = 0.9

                    accept_p = (ex_prob ** beta).clamp(min=p_floor, max=p_cap)
                    # accept_p = ex_prob
                    
                    if random_prob[vi] < accept_p:
                        # accept
                        accepted_tokens.append(int(curr_token))
                        # accepted_tokens.append(int(prev_prob))
                        if len(accepted_tokens) == W:
                            break
                        new_accepted_token += 1
                    else:
                        # reject
                        first_unmatched_token = vi
                        probe_ctx["reject_cnt"] += 1
                        break

                
                if first_unmatched_token == -1:
                    # all tokens are accepted
                    break
                else:
                    prev_token_prob = curr_token_probs[first_unmatched_token-1:].squeeze(0)
                    prev_token_ids = curr_token_ids[first_unmatched_token-1:].squeeze(0)
                    draft_input_ids = curr_token_ids[first_unmatched_token-1:-1].squeeze(0)

                window_size = W - len(accepted_tokens)
                start_pos = start_cnt + len(accepted_tokens) - 1
            
            probing["accept_length"] += new_accepted_token
                
            # Rollback delete false token cache
            for layer_idx in range(len(past.key_cache)):
                past.key_cache[layer_idx] = past.key_cache[layer_idx][..., :-window_size,:]
                past.value_cache[layer_idx] = past.value_cache[layer_idx][..., :-window_size,:]

            if len(accepted_tokens) == W:
                break

                
        for c in range(W):
            generated_tokens[r * W + c] = int(accepted_tokens[c])
 
        # accepted_tokens.insert(0, accepted_tokens.pop())
        draft_input_ids = torch.tensor(accepted_tokens, device=device)

        probe_ctx["forward_cnt"] += probing["forward_cnt"]
        probe_ctx["avg_accept_length"] += probing["accept_length"]

    probe_ctx["avg_accept_length"] = probe_ctx["avg_accept_length"] / probe_ctx["forward_cnt"]
    probe_ctx["accept_ratio"] = 1 - probe_ctx["reject_cnt"] / probe_ctx["verify_cnt"]
    print(probe_ctx)
    return generated_tokens

@torch.inference_mode()
def partial_row_parallel_logits(
    mm, proc, prompt, H, W,
    cfg_weight=5.0, temperature=1.0,
    row_batch_size: int = 8,
):
    device = next(mm.language_model.parameters()).device

    tok = proc.tokenizer
    ids = torch.LongTensor(tok.encode(prompt)).to("cuda")
    
    toks = torch.stack([ids, ids]).to("cuda")
    toks[1, 1:-1] = proc.pad_id

    inp = mm.language_model.get_input_embeddings()(toks)
    L_prefill = inp.shape[1]

    T = H * W
    gt_tokens, probs_all = [], []
    generated_tokens = torch.zeros((T), dtype=torch.int64, device="cuda")

    past = None
    pos_cur = L_prefill
    prev_row_embs = []
    
    x_row = 1

    g = torch.Generator(device="cuda")
    g.manual_seed(42)

    for c in range(W * x_row):
        out = mm.language_model.model(
            inputs_embeds=inp,
            use_cache=True,
            past_key_values=past,
        )
        hs, past = out.last_hidden_state, out.past_key_values
        last_h = hs[:, -1, :]                           # [2, D]
        log = cfg_merge(mm.gen_head(last_h), cfg_weight).squeeze(0) # [V]
        
        probs = F.softmax(log / temperature, dim=-1)
        probs_all.append(probs.squeeze().float().detach().cpu())

        nxt = torch.multinomial(probs, num_samples=1, generator=g).squeeze(0) # generator=g
        token_id = int(nxt)
        generated_tokens[0 * W + c] = token_id
        gt_tokens.append(token_id)

        both = torch.tensor([token_id, token_id], device="cuda").view(-1)
        img_emb = mm.prepare_gen_img_embeds(both)       # [2, D]
        prev_row_embs.append(img_emb[0].detach())       # [D]

        inp = img_emb.unsqueeze(1)                      # [2,1,D]
        pos_cur += 1


    prev_row_embs = prev_row_embs[-row_batch_size:]
    prev_row_embs.insert(0, prev_row_embs.pop())

    
    row_fwd_count = W // row_batch_size if W % row_batch_size == 0 else W // row_batch_size + 1
    for r in range(x_row, H):
        pos_base = L_prefill + r * W   
        
        for rf in range(row_fwd_count):
            row_cond = torch.stack(prev_row_embs, dim=0).to(device)          # [W, D]
            step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous() # [2, W, D]
            pos_ids = (torch.arange(row_batch_size, device=device, dtype=torch.long) + pos_base + rf * row_batch_size).unsqueeze(0).expand(2, -1)
            out_prop = mm.language_model.model(
                inputs_embeds=step_emb,
                past_key_values=past,
                position_ids=pos_ids,
                use_cache=True
            )
            past = out_prop.past_key_values
            log_q = cfg_merge(mm.gen_head(out_prop.last_hidden_state), cfg_weight).squeeze(0)
            q_probs  = F.softmax(log_q / (temperature), dim=-1)         # [W, V]
            
            proposal = torch.multinomial(q_probs, 1, generator=g).squeeze(-1)      # [W]
            prev_row_embs = list(mm.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0].unbind(dim=0))

            final_row = proposal.clone()
        
            for c in range(rf * row_batch_size, (rf + 1) * row_batch_size):
                generated_tokens[r * W + c] = int(final_row[c % row_batch_size])
            
            if rf == row_fwd_count - 1 and len(prev_row_embs) > 1:    
                prev_row_embs.insert(0, prev_row_embs.pop())

    return generated_tokens


@torch.inference_mode()
def row_parallel_experiments(
    mm, proc, prompt, H, W,
    cfg_weight=5.0, temperature=1.0,
    ar_row=1, rp_row=23, do_sample: bool = True,
): 
    device = next(mm.language_model.parameters()).device
    
    K_NEIGHBORS = 10
    USE_RANDOM_NEIGHBOR = False

    # dist_np = np.load("probe_runtime/embeddings_dist.npy")
    # dist = torch.from_numpy(dist_np)
    # neighbor_idx = torch.argsort(dist, dim=-1)
    # neighbor_idx_k = neighbor_idx[:, 1:K_NEIGHBORS+1]   
    # neighbors = neighbor_idx_k.to(device)          # [V, K]


    tok = proc.tokenizer
    ids = torch.LongTensor(tok.encode(prompt)).to("cuda")
    
    g = torch.Generator(device="cuda")
    g.manual_seed(42)
    
    toks = torch.stack([ids, ids]).to("cuda")
    toks[1, 1:-1] = proc.pad_id

    inp = mm.language_model.get_input_embeddings()(toks)
    L_prefill = inp.shape[1]

    T = H * W
    gt_tokens, probs_all = [], []
    generated_tokens = torch.zeros((T), dtype=torch.int64, device="cuda")

    past = None
    pos_cur = L_prefill
    prev_row_embs = []
    
    TOP_L, TOP_R = 10, 100
    
    
    for c in range(W * ar_row):
        out = mm.language_model.model(
            inputs_embeds=inp,
            use_cache=True,
            past_key_values=past,
        )
        hs, past = out.last_hidden_state, out.past_key_values
        last_h = hs[:, -1, :]                           # [2, D]
        log = cfg_merge(mm.gen_head(last_h), cfg_weight).squeeze(0) # [V]
        
        probs = torch.softmax(log / temperature, dim=-1)
        probs_all.append(probs.squeeze().float().detach().cpu())

        if do_sample:
            if True:
                nxt = torch.multinomial(probs, num_samples=1, generator=g).squeeze(0) # 
            if False:
                probs = probs.unsqueeze(0)
                top100 = probs.topk(TOP_R, dim=-1).indices              # [T,100]
                nxt = top100[torch.arange(probs.size(0)), torch.randint(TOP_L, TOP_R, (probs.size(0),))]
        else:
            nxt = probs.argmax(dim=-1, keepdim=True)
        token_id = int(nxt)
        generated_tokens[W * 0 + c] = token_id
        gt_tokens.append(token_id)

        both = torch.tensor([token_id, token_id], device="cuda").view(-1)
        img_emb = mm.prepare_gen_img_embeds(both)       # [2, D]
        prev_row_embs.append(img_emb[0].detach())       # [D]
        inp = img_emb.unsqueeze(1)                      # [2,1,D]
    
    def top_p_stability_verify(draft_ids, target_logits, p=0.9, temp=1.0):   
        probs = F.softmax(target_logits / temp, dim=-1) # [B, W, V]
        draft_probs = torch.gather(probs, -1, draft_ids.unsqueeze(-1)).squeeze(-1) # [B, W]
        sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)        
        cutoff_index = (cumsum_probs > p).float().argmax(dim=-1) # [B, W]
        cutoff_prob = torch.gather(sorted_probs, -1, cutoff_index.unsqueeze(-1)).squeeze(-1) # [B, W]
        is_stable = (draft_probs >= cutoff_prob)
        new_sampled_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(draft_ids.shape)
        final_ids = torch.where(is_stable, draft_ids, new_sampled_ids)
        convergence_rate = is_stable.float().mean().item()
        return final_ids, convergence_rate
    
    prev_row_embs = prev_row_embs[-W:]
    prev_row_embs.insert(0, prev_row_embs.pop())
    row_cond = torch.stack(prev_row_embs)             # [W, D]

    for r in range(ar_row, ar_row+rp_row):
        pos_base = L_prefill + r * W
        if False:
            cand_center = prev_row_embs
            cand_right = torch.roll(prev_row_embs, shifts=1, dims=0)
            cand_left = torch.roll(prev_row_embs, shifts=-1, dims=0)
            
            candidates_list = [cand_center, cand_right, cand_left]
            candidates_scores = []
            
            for cand_input in candidates_list:
                cand_input = torch.stack([cand_input, cand_input], dim=0).contiguous()
                pos_ids = (torch.arange(W, device=device, dtype=torch.long) + pos_base).unsqueeze(0).expand(2, -1)
                out_prop = mm.language_model.model(
                    inputs_embeds=cand_input,          # [2, W, D]
                    past_key_values=_to_cache(past),
                    position_ids=pos_ids,            # [2, W]
                    use_cache=True
                )
                cand_logits = cfg_merge(mm.gen_head(out_prop.last_hidden_state), cfg_weight).squeeze(0)
                cand_probs  = F.softmax(cand_logits / temperature, dim=-1)         # [W, V]
                candidates_scores.append(cand_probs)
            
            stacked_probs = torch.stack(candidates_scores, dim=0) 
            if True:
                mixed_probs = torch.mean(stacked_probs, dim=0)
                draft_token_ids = torch.multinomial(mixed_probs, 1).squeeze(-1)
            if False:
                epsilon = 1e-8
                entropy = -torch.sum(stacked_probs * torch.log(stacked_probs + epsilon), dim=-1)
                weights = F.softmax(-entropy, dim=0)
                weighted_probs = torch.sum(stacked_probs * weights.unsqueeze(-1), dim=0)
                draft_token_ids = torch.argmax(weighted_probs, dim=-1)
            if False:
                confidence_scores, _ = torch.max(stacked_probs, dim=-1) # [3, W]
                best_source_idx = torch.argmax(confidence_scores, dim=0) # [W]
                W, V = stacked_probs.shape[1], stacked_probs.shape[2]
                gather_idx = best_source_idx.view(1, W, 1).expand(1, W, V)
                chosen_probs = torch.gather(stacked_probs, 0, gather_idx).squeeze(0) # [W, V]
                draft_token_ids = torch.argmax(chosen_probs, dim=-1)
            if False:
                cand_ids = [torch.argmax(p, dim=-1) for p in candidates_scores] 
                stacked_ids = torch.stack(cand_ids, dim=0) # [3, W]
                mode_values, _ = torch.mode(stacked_ids, dim=0) 
                draft_token_ids = mode_values
            if False:
                source_scores, _ = torch.max(stacked_probs, dim=-1) # [3, W]
                source_selection_probs = F.softmax(source_scores / 0.5, dim=0) # [3, W], 温度控制区分度
                selected_source = torch.multinomial(source_selection_probs.permute(1, 0), 1).squeeze(-1) # [W]
                cand_ids = torch.stack([torch.argmax(p, dim=-1) for p in candidates_scores], dim=0) # [3, W]
                draft_token_ids = torch.gather(cand_ids, 0, selected_source.unsqueeze(0)).squeeze(0)
                
            row_cond = mm.prepare_gen_img_embeds(draft_token_ids)
        
        
        for _ in range(1):
            step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous() # [2, W, D]
            step_emb = 0.5 * step_emb
            pos_ids = (torch.arange(W, device=device, dtype=torch.long) + pos_base).unsqueeze(0).expand(2, -1)
            out_prop = mm.language_model.model(
                inputs_embeds=step_emb,          # [2, W, D]
                past_key_values=_to_cache(past),
                # attention_mask=attn_mask,        # [2, 1, W, T_k+W], 0/-inf
                position_ids=pos_ids,            # [2, W]
                use_cache=True
            )

            log_q = cfg_merge(mm.gen_head(out_prop.last_hidden_state), cfg_weight).squeeze(0)
            q_probs  = F.softmax(log_q / temperature, dim=-1)         # [W, V]
            if _>=1:
                proposal = torch.multinomial(q_probs, 1).squeeze(-1)      # [W]
            else:
                proposal = q_probs.argmax(dim=-1)
            prev_row_embs = list(mm.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0].unbind(dim=0))
            prev_row_embs.insert(0, prev_row_embs.pop())
            row_cond = torch.stack(prev_row_embs, dim=0).to(device)          # [W, D]
        
        MAX_ITER = 0
        prev_ids = None
        for i in range(MAX_ITER):
            step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous() # [2, W, D]
            pos_ids = (torch.arange(W, device=device, dtype=torch.long) + pos_base).unsqueeze(0).expand(2, -1)
            out_prop = mm.language_model.model(
                inputs_embeds=step_emb,          # [2, W, D]
                past_key_values=_to_cache(past),
                position_ids=pos_ids,            # [2, W]
                use_cache=True
            )
            log_q = cfg_merge(mm.gen_head(out_prop.last_hidden_state), cfg_weight).squeeze(0)
            if i > 0:
                current_temp = temperature * (1.0 - i * 0.1)
                proposal, cr = top_p_stability_verify(prev_ids, log_q, 0.75, current_temp)
                # print("Iteration: ", i, ", stability: ", cr)
            else:
                q_probs  = F.softmax(log_q / temperature, dim=-1)   
                proposal = torch.multinomial(q_probs, 1).squeeze(-1)  
            prev_ids = proposal
            prev_row_embs = list(mm.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0].unbind(dim=0))
            prev_row_embs.insert(0, prev_row_embs.pop())
            row_cond = torch.stack(prev_row_embs, dim=0).to(device)          # [W, D]
        
        step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous() # [2, W, D]
        pos_ids = (torch.arange(W, device=device, dtype=torch.long) + pos_base).unsqueeze(0).expand(2, -1)
        out_prop = mm.language_model.model(
            inputs_embeds=step_emb,
            past_key_values=past,
            position_ids=pos_ids,            # [2, W]
            use_cache=True
        )

        log_q = cfg_merge(mm.gen_head(out_prop.last_hidden_state), cfg_weight).squeeze(0)
        q_probs  = F.softmax(log_q / temperature, dim=-1)
        proposal = torch.multinomial(q_probs, 1).squeeze(-1)
        prev_row_embs = list(mm.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0].unbind(dim=0))
        prev_row_embs.insert(0, prev_row_embs.pop())
        row_cond = torch.stack(prev_row_embs, dim=0).to(device)          # [W, D]
        
        # step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous()
        # pos_ids = (torch.arange(W, device=device, dtype=torch.long) + pos_base).unsqueeze(0).expand(2, -1)
        # out_prop = mm.language_model.model(
        #     inputs_embeds=step_emb,
        #     past_key_values=past,
        #     position_ids=pos_ids,
        #     use_cache=True
        # )
        # past = out_prop.past_key_values
        # log_q = cfg_merge(mm.gen_head(out_prop.last_hidden_state), cfg_weight).squeeze(0)

        # q_probs  = F.softmax(log_q / (temperature), dim=-1)   
        # for prob in q_probs:
        #     probs_all.append(prob.float().detach().cpu())
        # if do_sample:      # [W, V]
        #     if True:
        #         proposal = torch.multinomial(q_probs, 1, generator=g).squeeze(-1)      # [W]
        #     if False:
        #         top100 = q_probs.topk(TOP_R, dim=-1).indices              # [T, 100]
        #         proposal = top100[torch.arange(q_probs.size(0)), torch.randint(TOP_L, TOP_R, (q_probs.size(0),))]
        #         if USE_RANDOM_NEIGHBOR:
        #             rand_idx = torch.randint(
        #                 0,
        #                 neighbors.size(1),
        #                 (proposal.size(0),),
        #                 generator=g,
        #                 device=q_probs.device,
        #             )                                                                # [W]
        #             proposal = neighbors[proposal, rand_idx]                         # [W]
        #         else:
        #             proposal = neighbors[proposal, 0]        
        # else:
        #     proposal = q_probs.argmax(dim=-1)
        # prev_row_embs = mm.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0]
        # row_cond = torch.roll(prev_row_embs, shifts=1, dims=0)
        
        final_row = proposal.clone()
        for c in range(W):
            generated_tokens[r * W + c] = int(final_row[c])
        # prev_row_embs.insert(0, prev_row_embs.pop())

    inp = prev_row_embs[0].unsqueeze(0)
    inp = torch.stack([inp, inp], dim=0)
    for c in range((ar_row+rp_row)*W, H*W):
        out = mm.language_model.model(
            inputs_embeds=inp,
            use_cache=True,
            past_key_values=past,
        )
        hs, past = out.last_hidden_state, out.past_key_values
        last_h = hs[:, -1, :]                           # [2, D]
        log = cfg_merge(mm.gen_head(last_h), cfg_weight).squeeze(0) # [V]
        
        probs = torch.softmax(log / temperature, dim=-1)
        probs_all.append(probs.squeeze().float().detach().cpu())
        if do_sample:
            nxt = torch.multinomial(probs, num_samples=1, generator=g).squeeze(0) # 
        else:
            nxt = probs.argmax(dim=-1, keepdim=True)
        token_id = int(nxt)
        generated_tokens[c] = token_id
        gt_tokens.append(token_id)

        both = torch.tensor([token_id, token_id], device="cuda").view(-1)
        img_emb = mm.prepare_gen_img_embeds(both)       # [2, D]
        inp = img_emb.unsqueeze(1)                      # [2,1,D]
    

    return generated_tokens


from contextlib import contextmanager
@contextmanager
def no_causal_mask_once(llama_model):
    m = llama_model
    old_flags = [blk.self_attn.is_causal for blk in m.layers]
    old_upd = m._update_causal_mask

    try:
        for blk in m.layers:
            blk.self_attn.is_causal = False

        def _no_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions):
            return None

        m._update_causal_mask = _no_mask
        yield
    finally:
        for blk, flag in zip(m.layers, old_flags):
            blk.self_attn.is_causal = flag
        m._update_causal_mask = old_upd

if __name__ == "__main__":
    gen_kwargs = dict(
        parallel_size=1,
        image_token_num_per_image=image_token_num_per_image,
        img_size=target_size,
        patch_size=patch_size,
    )
    save_dir = "parallel_test"
    probe_dir = "probe_runtime"
    images = []
    baseline_prob, row_parallel_prob = [], []
    baseline_token, row_parallel_token = [], []
    for idx, desc in enumerate(q_image_content_conditions):
        if idx > 4:
            break
            pass
        prompt = build_prompt(desc)
        base = slugify_first_words(desc, n=6)

        t0 = time.perf_counter()
        
        # generated_tokens, bs_probs_all = baseline_gt_logits(vl_gpt, vl_chat_processor, prompt, H, W, cfg_weight=3.0, temperature=1.0, do_sample=True)
        # baseline_token.append(generated_tokens.detach().cpu().numpy())
        # generated_tokens, pl_probs_all = baseline_row_parallel_logits(vl_gpt, vl_chat_processor, prompt, H, W, cfg_weight=3.0, temperature=1.0, ar_row=1, do_sample=True)
        # row_parallel_token.append(generated_tokens.detach().cpu().numpy())
       
        # baseline_prob.append(bs_probs_all)
        # row_parallel_prob.append(pl_probs_all)

        # generated_tokens = spec_row_parallel_logits(vl_gpt, vl_chat_processor, prompt, H, W, row_parallel=True, cfg_weight=3.0, temperature=1.0)
        # generated_tokens = partial_row_parallel_logits(vl_gpt, vl_chat_processor, prompt, H, W, cfg_weight=3.0, temperature=1.0, row_batch_size=24)
        
        generated_tokens = row_parallel_experiments(vl_gpt, vl_chat_processor, prompt, H, W, cfg_weight=3.0, temperature=1.0, ar_row=1, rp_row=23,do_sample=True)
        dt = time.perf_counter() - t0
        
        img_size = 384
        with torch.no_grad():
            dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[1, 8, img_size//patch_size, img_size//patch_size])
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

            dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

            visual_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec

            os.makedirs(save_dir, exist_ok=True)

            name = "img"
            save_path = os.path.join(save_dir, f"{name}_{idx}.jpg")
            PIL.Image.fromarray(dec[0]).save(save_path)
            images.append(dec[0])
            
            print(f"[{idx}] {base}  ->  {dt:.2f}s  | ") #saved: {paths[0]}")
    
    if len(images) > 0:
        datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        row = np.concatenate(images, axis=1)
        save_path = os.path.join(save_dir, f"test_{datetime}.jpg")
        PIL.Image.fromarray(row).save(save_path)
    
    if False:
        np.save(os.path.join(probe_dir, "baseline_prob.npy"), np.array(baseline_prob, dtype=np.float32))
        np.save(os.path.join(probe_dir, "row_parallel_prob.npy"), np.array(row_parallel_prob, dtype=np.float32))
        np.save(os.path.join(probe_dir, "baseline_token.npy"), np.array(baseline_token, dtype=np.int32))
        np.save(os.path.join(probe_dir, "row_parallel_token.npy"), np.array(row_parallel_token, dtype=np.int32))

        # torch.cuda.empty_cache()

    if False:
        np.save(os.path.join(probe_dir, "baseline_prob_sample20.npy"), np.array(baseline_prob, dtype=np.float32))
        np.save(os.path.join(probe_dir, "row_parallel_prob_sample20.npy"), np.array(row_parallel_prob, dtype=np.float32))
        np.save(os.path.join(probe_dir, "baseline_token_sample20.npy"), np.array(baseline_token, dtype=np.int32))
        np.save(os.path.join(probe_dir, "row_parallel_token_sample20.npy"), np.array(row_parallel_token, dtype=np.int32))