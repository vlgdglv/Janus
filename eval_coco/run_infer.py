import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pathlib import Path
import torch
from PIL import Image
import open_clip
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from typing import List
import numpy as np
import torch.nn.functional as F
import threading, queue

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def compute_clip_score(
    image_paths,
    texts,
    model_name: str = "ViT-L-14",
    pretrained: str = None,
    batch_size: int = 64,
    device: str | None = None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    assert len(image_paths) == len(texts), "image_paths and texts must have same length"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device).eval()

    sims = []
    N = len(texts)

    for start in tqdm(range(0, N, batch_size), total=N // batch_size):
        end = min(start + batch_size, N)
        batch_imgs = []
        for p in image_paths[start:end]:
            p = Path(p)
            img = Image.open(p).convert("RGB")
            batch_imgs.append(preprocess(img))

        images = torch.stack(batch_imgs).to(device)
        text_batch = texts[start:end]
        text_tokens = tokenizer(text_batch).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(images)
            txt_feat = model.encode_text(text_tokens)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        batch_sims = (img_feat * txt_feat).sum(dim=-1) 
        sims.append(batch_sims.cpu())

    sims = torch.cat(sims)
    return float(sims.mean()), float(sims.std())

def calc_original_clip_score(
    model_name: str = "ViT-L-14",
    pretrained: str = None,
    batch_size: int = 64,
    device: str | None = None,
):
    with open("eval_coco/data/coco2017_val_prompts.json", "r") as f:
        prompts = json.load(f)

    image_dir = Path("~/Data/coco2017/val2017").expanduser()
    texts = [p["caption"] for p in prompts]
    image_paths = [image_dir / p["file_name"] for p in prompts]

    mean, std = compute_clip_score(image_paths, texts, model_name, pretrained, batch_size, device)
    print("mean: {:.4f}, std: {:.4f}".format(mean, std))

def build_prompt(desc: str) -> str:
    conv = [{"role": "User", "content": desc}, {"role": "Assistant", "content": ""}]
    s = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conv,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return s + vl_chat_processor.image_start_tag

def batched_generate_tokens(
    mm,
    proc,
    prompts_texts: List[str],
    H: int,
    W: int,
    cfg_weight: float = 3.0,
    temperature: float = 1.0,
    device: str = "cuda",
    return_hidden_states: bool = False
):
    tok = proc.tokenizer
    device = torch.device(device)
    
    encoded = [tok.encode(build_prompt(p)) for p in prompts_texts]

    B = len(encoded)
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long, device=device)
    base_len = torch.cat([lengths, lengths], dim=0)         
    max_len = max(len(e) for e in encoded)
    
    pad_id = proc.pad_id
    full_token = []
    ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        e_tensor = torch.tensor(e, dtype=torch.long, device=device)
        full_token.append(e_tensor)
        ids[i, : len(e)] = e_tensor

    attn_mask = (ids != pad_id).long()  

    cond_ids = ids
    uncond_ids = ids.clone()
    if max_len > 2:
        uncond_ids[:, 1:-1] = pad_id

    base_pos = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, L_i in enumerate(lengths.tolist()):
        base_pos[i, :L_i] = torch.arange(L_i, device=device)
        if L_i < max_len:
            base_pos[i, L_i:] = max(L_i - 1, 0)
    pos_ids = torch.cat([base_pos, base_pos], dim=0) 
 

    toks = torch.cat([cond_ids, uncond_ids], dim=0) 
    attn_mask = torch.cat([attn_mask, attn_mask], dim=0)
    inp = mm.language_model.get_input_embeddings()(toks)

    T = H * W
    generated_tokens = torch.zeros((B, T), dtype=torch.int, device=device)
    if return_hidden_states:
        all_step_hidden_states = []

    past = None
    for t in range(T):
        out = mm.language_model.model(
            inputs_embeds=inp,
            attention_mask=attn_mask,
            use_cache=True,
            past_key_values=past,
            position_ids=pos_ids, 
        )
        past = out.past_key_values
        hs = out.last_hidden_state[:, -1, :] # [2B, D]

        if return_hidden_states:
            all_step_hidden_states.append(hs.detach().cpu())
        
        log_all = mm.gen_head(hs)

        V = log_all.size(-1)
        log_all = log_all.view(2, B, V)
        cond_log = log_all[0]       # [B, V]
        uncond_log = log_all[1]     # [B, V]
        log = uncond_log + cfg_weight * (cond_log - uncond_log)  # [B, V]

        probs = torch.softmax(log / temperature, dim=-1)
        
        nxt = torch.multinomial(probs, num_samples=1)      # [B, 1]
        nxt_flat = nxt.view(-1)                            # [B]

        generated_tokens[:, t] = nxt_flat

        both = torch.cat([nxt_flat, nxt_flat], dim=0)      # [2B]
        img_emb = mm.prepare_gen_img_embeds(both)          # [2B, D]
        inp = img_emb.unsqueeze(1)                         # [2B, 1, D]

        ones = torch.ones(attn_mask.size(0), 1,
                      dtype=attn_mask.dtype,
                      device=attn_mask.device)
        attn_mask = torch.cat([attn_mask, ones], dim=1)
        
        pos_ids = (base_len + t).unsqueeze(1)            # [2B, 1]
    
    for i in range(B):
        full_token[i] = torch.cat([full_token[i], generated_tokens[i]], dim=0)

    if return_hidden_states:
        all_step_hidden_states = torch.stack(all_step_hidden_states, dim=1) # [2B, T, D]
        return generated_tokens, all_step_hidden_states.half()
    else:
        return generated_tokens, full_token



def cfg_merge(log_all, cfg_weight=5.0):
    return log_all[0:1] + cfg_weight * (log_all[0:1] - log_all[1:2])

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

def row_parallel_generate_tokens(
    mm,
    proc,
    prompts_text: List[str] | str,
    H: int,
    W: int,
    cfg_weight: float = 3.0,
    temperature: float = 1.0,
    device: str = "cuda",
    x_more_fwd: int = 0
):
    device = next(mm.language_model.parameters()).device

    tok = proc.tokenizer
    ids = torch.LongTensor(tok.encode(prompts_text)).to("cuda")
    
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

        nxt = torch.multinomial(probs, num_samples=1).squeeze(0) # generator=g
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

    for r in range(x_row, H):
        T_k = L_prefill + r * W
        pos_base = T_k      
        K_total = T_k + W

        for _ in range(x_more_fwd):
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
            proposal = torch.multinomial(q_probs, 1).squeeze(-1)      # [W]
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
        q_probs  = F.softmax(log_q / (temperature), dim=-1)         # [W, V]
        
        for log in log_q:
            probs_all.append(log.float().detach().cpu())
        proposal = torch.multinomial(q_probs, 1).squeeze(-1)      # [W]
        prev_row_embs = list(mm.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0].unbind(dim=0))

        final_row = proposal.clone()
    
        for c in range(W):
            generated_tokens[r * W + c] = int(final_row[c])

        prev_row_embs.insert(0, prev_row_embs.pop())

    return generated_tokens

def row_parallel_spec_generate_tokens(
    mm,
    proc,
    prompts_text: List[str] | str,
    H: int,
    W: int,
    cfg_weight: float = 3.0,
    temperature: float = 1.0,
    device: str = "cuda",
    x_more_fwd: int = 0
):
    tok = proc.tokenizer
    ids = torch.LongTensor(tok.encode(prompts_text)).to("cuda")
    
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
                    ex_prob = prev_prob / curr_prob 
                    
                    beta = 0.1
                    p_floor = 0.1
                    p_cap   = 0.9

                    # accept_p = (ex_prob ** beta).clamp(min=p_floor, max=p_cap)
                    accept_p = ex_prob
                    
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

        # probe_ctx["forward_cnt"] += probing["forward_cnt"]
        # probe_ctx["avg_accept_length"] += probing["accept_length"]

    # probe_ctx["avg_accept_length"] = probe_ctx["avg_accept_length"] / probe_ctx["forward_cnt"]
    # probe_ctx["accept_ratio"] = 1 - probe_ctx["reject_cnt"] / probe_ctx["verify_cnt"]
    # print(probe_ctx)
    return generated_tokens


def generate_coco_baseline_batched(
    mm,
    proc,
    all_prompts: List[str],
    H: int,
    W: int,
    img_size: int,
    patch_size: int,
    batch_prompts: int = 4,
    cfg_weight: float = 3.0,
    temperature: float = 1.0,
    device: str = "cuda", 
    save_dir: str = "./results",
    save_token_only: bool = False
):  
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)
    mm = mm.to(device).to(torch.float16).eval()

    mm.gen_vision_model.to("cpu")
    global_idx = 0
    with torch.no_grad():
        N = len(all_prompts)
        all_images = []
        for i in tqdm(range(0, len(all_prompts), batch_prompts), total=N // batch_prompts):
            sub_prompts = all_prompts[i : i + batch_prompts]
            """
                sub_prompts: {'image_id': int, 'file_name': str, 'caption': str}
            """

            prompts_text = [p["caption"] for p in sub_prompts]
            image_name = [p["image_id"] for p in sub_prompts]
            gen_tokens, full_tokens = batched_generate_tokens(
                mm,
                proc,
                prompts_text,
                H,
                W,
                cfg_weight=cfg_weight,
                temperature=temperature,
                device=device,
            )
        
            B, T = gen_tokens.shape
            if save_token_only:
                cnter = 0
                for tokens in full_tokens:
                    sample_idx = i + cnter 
                    cnter += 1 
                    save_path = Path(save_dir) / f"tokens_{sample_idx}.pt"
                    torch.save(tokens, save_path)
            else:
                # Off load
                mm.language_model.to("cpu")
                mm.gen_vision_model.to(device)
                for b in range(B):
                    tokens_b = gen_tokens[b]
                    tokens_b_int = tokens_b.to(dtype=torch.int).unsqueeze(0) 
                    dec = mm.gen_vision_model.decode_code(
                        tokens_b_int,
                        shape=[1, 8, img_size // patch_size, img_size // patch_size],
                    )
                    dec = (
                        dec.to(torch.float32)
                        .cpu()
                        .numpy()
                        .transpose(0, 2, 3, 1)
                    )  # [1, H', W', C]

                    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
                    img_np = dec[0]  # [H', W', 3]
                    save_path = Path(save_dir) / f"generated_{image_name[b]}.jpg"
                    Image.fromarray(img_np).save(save_path)

                    all_images.append(img_np)

                    global_idx += 1
                mm.language_model.to(device)
                mm.gen_vision_model.to("cpu")
            
    print(f"\nDone. Saved {global_idx} things to {save_dir}")



def generate_coco_baseline_batched_with_overlap(
    mm,
    proc,
    all_prompts: List[str],
    H: int,
    W: int,
    img_size: int,
    patch_size: int,
    batch_prompts: int = 4,
    cfg_weight: float = 3.0,
    temperature: float = 1.0,
    device: str = "cuda", 
    save_dir: str = "./results",
    save_token_only: bool = False
):  
    def writer_worker(q: "queue.Queue",):
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break

            tokens_cpu, save_path = item
            tokens_cpu = tokens_cpu.to(dtype=torch.int32)
            torch.save(tokens_cpu, save_path)
            q.task_done()

    save_q = queue.Queue(maxsize=256)
    t = threading.Thread(target=writer_worker, args=(save_q,), daemon=True)
    t.start()

    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)
    mm = mm.to(device).to(torch.float16).eval()

    mm.gen_vision_model.to("cpu")
    global_idx = 0
    with torch.no_grad():
        N = len(all_prompts)
        all_images = []
        for i in tqdm(range(0, len(all_prompts), batch_prompts), total=N // batch_prompts):
            sub_prompts = all_prompts[i : i + batch_prompts]
            """
                sub_prompts: {'image_id': int, 'file_name': str, 'caption': str}
            """

            prompts_text = [p["caption"] for p in sub_prompts]
            image_name = [p["image_id"] for p in sub_prompts]
            gen_tokens, full_tokens = batched_generate_tokens(
                mm,
                proc,
                prompts_text,
                H,
                W,
                cfg_weight=cfg_weight,
                temperature=temperature,
                device=device,
            )
        
            B, T = gen_tokens.shape
            if save_token_only:
                cnter = 0
                for tokens in full_tokens:
                    sample_idx = i + cnter 
                    cnter += 1 
                    save_path = str(Path(save_dir) / f"tokens_{sample_idx}.pt")
                    tokens_cpu = tokens.detach().to("cpu", non_blocking=True)
                    save_q.put((tokens_cpu, save_path))
            else:
                # Off load
                mm.language_model.to("cpu")
                mm.gen_vision_model.to(device)
                for b in range(B):
                    tokens_b = gen_tokens[b]
                    tokens_b_int = tokens_b.to(dtype=torch.int).unsqueeze(0) 
                    dec = mm.gen_vision_model.decode_code(
                        tokens_b_int,
                        shape=[1, 8, img_size // patch_size, img_size // patch_size],
                    )
                    dec = (
                        dec.to(torch.float32)
                        .cpu()
                        .numpy()
                        .transpose(0, 2, 3, 1)
                    )  # [1, H', W', C]

                    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
                    img_np = dec[0]  # [H', W', 3]
                    save_path = Path(save_dir) / f"generated_{image_name[b]}.jpg"
                    Image.fromarray(img_np).save(save_path)

                    all_images.append(img_np)

                mm.language_model.to(device)
                mm.gen_vision_model.to("cpu")

            global_idx += 1

    save_q.put(None)
    save_q.join()
    t.join()
    print(f"\nDone. Saved {global_idx} things to {save_dir}")

def generate_coco_row_parallel(
    mm,
    proc,
    all_prompts: List[str],
    H: int,
    W: int,
    img_size: int,
    patch_size: int,
    batch_prompts: int = 4,
    cfg_weight: float = 3.0,
    temperature: float = 1.0,
    device: str = "cuda", 
    save_dir: str = "./results",
    x_more_fwd: int = 0
):  
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)
    mm = mm.to(device).to(torch.bfloat16).eval()

    global_idx = 0
    with torch.no_grad():
        N = len(all_prompts)
        for prompts in tqdm(all_prompts, total=N):
            """
                prompts: {'image_id': int, 'file_name': str, 'caption': str}
            """

            prompts_text = prompts["caption"]
            image_name = prompts["image_id"]
            gen_tokens = row_parallel_generate_tokens(
                mm,
                proc,
                prompts_text,
                H,
                W,
                cfg_weight=cfg_weight,
                temperature=temperature,
                device=device,
                x_more_fwd=x_more_fwd
            )
        
            dec = vl_gpt.gen_vision_model.decode_code(gen_tokens.to(dtype=torch.int), shape=[1, 8, img_size//patch_size, img_size//patch_size])
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

            dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

            img_np = dec[0]  # [H', W', 3]
            save_path = Path(save_dir) / f"generated_{image_name}.jpg"
            Image.fromarray(img_np).save(save_path)



def generate_coco_row_parallel_spec(
    mm,
    proc,
    all_prompts: List[str],
    H: int,
    W: int,
    img_size: int,
    patch_size: int,
    batch_prompts: int = 4,
    cfg_weight: float = 3.0,
    temperature: float = 1.0,
    device: str = "cuda", 
    save_dir: str = "./results",
    x_more_fwd: int = 0
):  
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)
    mm = mm.to(device).to(torch.bfloat16).eval()

    global_idx = 0
    with torch.no_grad():
        N = len(all_prompts)
        for prompts in tqdm(all_prompts, total=N):
            """
                prompts: {'image_id': int, 'file_name': str, 'caption': str}
            """

            prompts_text = prompts["caption"]
            image_name = prompts["image_id"]
            gen_tokens = row_parallel_spec_generate_tokens(
                mm,
                proc,
                prompts_text,
                H,
                W,
                cfg_weight=cfg_weight,
                temperature=temperature,
                device=device,
                x_more_fwd=x_more_fwd
            )
        
            dec = vl_gpt.gen_vision_model.decode_code(gen_tokens.to(dtype=torch.int), shape=[1, 8, img_size//patch_size, img_size//patch_size])
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

            dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

            img_np = dec[0]  # [H', W', 3]
            save_path = Path(save_dir) / f"generated_{image_name}.jpg"
            Image.fromarray(img_np).save(save_path)
            
def generate_coco_statistics(
    mm,
    proc,
    all_prompts: List[str],
    H: int,
    W: int,
    batch_prompts: int = 4,
    cfg_weight: float = 3.0,
    temperature: float = 1.0,
    device: str = "cuda", 
    save_dir: str = "./stats_results",
    vocab_size: int = 16384,
):  
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)
    
    mm = mm.to(device).to(torch.bfloat16).eval()
    
    if vocab_size is None:
        try:
            vocab_size = mm.config.vocab_size 
        except:
            print("Warning: Could not infer vocab_size, defaulting to 16384.")
            vocab_size = 16384

    print(f"Initializing statistics matrices with Vocab Size: {vocab_size}")
    
    count_lr = np.zeros((vocab_size, vocab_size), dtype=np.float32)  # Left -> Right
    count_ud = np.zeros((vocab_size, vocab_size), dtype=np.float32)  # Up -> Down
    
    global_idx = 0
    
    if hasattr(mm, 'gen_vision_model'):
        mm.gen_vision_model.to("cpu")

    with torch.no_grad():
        N = len(all_prompts)
        
        for i in tqdm(range(0, len(all_prompts), batch_prompts), total=N // batch_prompts):
            sub_prompts = all_prompts[i : i + batch_prompts]
            prompts_text = [p["caption"] for p in sub_prompts]
            
            gen_tokens = batched_generate_tokens(
                mm,
                proc,
                prompts_text,
                H,
                W,
                cfg_weight=cfg_weight,
                temperature=temperature,
                device=device,
            )
            
            # [B, T] -> [B, H, W]
            tokens_np = gen_tokens.cpu().numpy()
            B = tokens_np.shape[0]
            grids = tokens_np.reshape(B, H, W)
            
            for b in range(B):
                g = grids[b] # [H, W]

                # --- 统计左右 (Left -> Right) ---
                # 取前 W-1 列作为 source，后 W-1 列作为 target
                left  = g[:, :-1].reshape(-1) 
                right = g[:,  1:].reshape(-1)
                
                # 过滤掉可能的 padding idx (如果有的话，通常 image token 不会有 padding)
                # 假设 valid tokens < vocab_size
                mask_lr = (left < vocab_size) & (right < vocab_size)
                np.add.at(count_lr, (left[mask_lr], right[mask_lr]), 1)

                # --- 统计上下 (Up -> Down) ---
                # 取前 H-1 行作为 source，后 H-1 行作为 target
                up   = g[:-1, :].reshape(-1)    
                down = g[1:,  :].reshape(-1)
                
                mask_ud = (up < vocab_size) & (down < vocab_size)
                np.add.at(count_ud, (up[mask_ud], down[mask_ud]), 1)

            global_idx += B

    print(f"Statistics collection finished. Processed {global_idx} samples.")

    print("Calculating probabilities and saving...")
    
    eps = 1e-12
    # P(Right | Left)
    row_sum_lr = count_lr.sum(axis=1, keepdims=True) + eps
    P_lr = count_lr / row_sum_lr
    
    # P(Down | Up)
    row_sum_ud = count_ud.sum(axis=1, keepdims=True) + eps
    P_ud = count_ud / row_sum_ud
    
    save_path = Path(save_dir) / "coco_token_stats.npz"
    np.savez_compressed(
        save_path,
        count_lr=count_lr,
        count_ud=count_ud,
        P_lr=P_lr,
        P_ud=P_ud
    )
    
    print(f"\nDone. Saved statistics to {save_path}")
    
    sparsity_lr = (count_lr > 0).mean()
    sparsity_ud = (count_ud > 0).mean()
    print(f"Matrix Sparsity Report:")
    print(f"Left-Right Transitions Non-zero: {sparsity_lr:.4%} (Specific tokens usually follow specific tokens)")
    print(f"Up-Down Transitions Non-zero:    {sparsity_ud:.4%}")


def generate_coco_hidden_states(
    mm,
    proc,
    all_prompts: List[str],
    H: int,
    W: int,
    batch_prompts: int = 4,
    cfg_weight: float = 3.0,
    temperature: float = 1.0,
    device: str = "cuda", 
    save_dir: str = "./hidden_state_results",
    vocab_size: int = 16384,
):  
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)
    
    mm = mm.to(device).to(torch.bfloat16).eval()
    
    if vocab_size is None:
        try:
            vocab_size = mm.config.vocab_size 
        except:
            print("Warning: Could not infer vocab_size, defaulting to 16384.")
            vocab_size = 16384

    print(f"Record hidden states and logits.")
    
    global_idx = 0
    
    if hasattr(mm, 'gen_vision_model'):
        mm.gen_vision_model.to("cpu")

    with torch.no_grad():
        N = len(all_prompts)
        
        for i in tqdm(range(0, len(all_prompts), batch_prompts), total=N // batch_prompts):
            sub_prompts = all_prompts[i : i + batch_prompts]
            prompts_text = [p["caption"] for p in sub_prompts]
            
            gen_tokens, batch_hs = batched_generate_tokens(
                mm,
                proc,
                prompts_text,
                H,
                W,
                cfg_weight=cfg_weight,
                temperature=temperature,
                device=device,
                return_hidden_states=True
            )
            
            save_path = os.path.join(save_dir, f"hs_shard{i}.pt")
            torch.save(batch_hs, save_path)


            
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_paths", type=str, required=True)
    # parser.add_argument("--prompt_path", type=str, required=True)

    # args = parser.parse_args()

    # with open(args.text_path, "r") as f:
    #     texts = json.load(f)

    # with open(args.image_paths, "r") as f:
    #     image_paths = json.load(f)

    
    # calc_original_clip_score(model_name="local-dir:/home/vlgd/Models/vit_large_patch14_clip_224.openai")
    
    model_path = "/home/vlgd/Models/Janus-Pro-1B/"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )

    # mode = "val"
    mode = "train"

    with open(f"eval_coco/data/coco2017_{mode}_prompts.json", "r") as f:
        all_prompts = json.load(f)
    
    # print(all_prompts[0 : 0 + 16])
    N_SAMPLE = 5000
    # N_SAMPLE = len(all_prompts)
    all_prompts = all_prompts[:N_SAMPLE]
    if True:
        bs = 24
        generate_coco_baseline_batched_with_overlap(
            vl_gpt,
            vl_chat_processor,
            all_prompts,
            H=24,
            W=24,
            img_size=384,
            patch_size=16,
            cfg_weight=3.0,
            batch_prompts=bs,
            temperature=1.0,
            device="cuda",
            save_dir="./generated/baseline_coco2017_{}_bs{}".format(mode, bs),
            save_token_only=True
        )
    if False:
        bs = 24
        generate_coco_baseline_batched(
            vl_gpt,
            vl_chat_processor,
            all_prompts,
            H=24,
            W=24,
            img_size=384,
            patch_size=16,
            cfg_weight=3.0,
            batch_prompts=bs,
            temperature=1.0,
            device="cuda",
            save_dir="./generated/baseline_coco2017_{}_bs{}".format(mode, bs),
            save_token_only=True
        )
    if False:
        generate_coco_row_parallel_batched(
            vl_gpt,
            vl_chat_processor,
            all_prompts,
            H=24,
            W=24,
            img_size=384,
            patch_size=16,
            cfg_weight=3.0,
            batch_prompts=16,
            temperature=1.0,
            device="cuda",
            save_dir="./generated/row_parallel_coco2017_val",
        )

    if False:
        generate_coco_row_parallel(
            vl_gpt,
            vl_chat_processor,
            all_prompts,
            H=24,
            W=24,
            img_size=384,
            patch_size=16,
            cfg_weight=3.0,
            batch_prompts=16,
            temperature=1.0,
            device="cuda",
            save_dir="./generated/row_parallel_coco2017_val_3_more_fwd",
            x_more_fwd=3
        )
        
    if False:
        generate_coco_row_parallel_spec(
            vl_gpt,
            vl_chat_processor,
            all_prompts,
            H=24,
            W=24,
            img_size=384,
            patch_size=16,
            cfg_weight=3.0,
            batch_prompts=16,
            temperature=1.0,
            device="cuda",
            save_dir="./generated/row_parallel_spec_coco2017_val",
            x_more_fwd=3
        )

    if False:
        generate_coco_statistics(
            vl_gpt,
            vl_chat_processor,
            all_prompts,
            H=24,
            W=24,
            cfg_weight=3.0,
            batch_prompts=16,
            temperature=1.0,
            device="cuda",
            save_dir="probe_runtime/",
        )

    if False:
        generate_coco_hidden_states(
            vl_gpt,
            vl_chat_processor,
            all_prompts,
            H=24,
            W=24,
            cfg_weight=3.0,
            batch_prompts=16,
            temperature=1.0,
            device="cuda",
            save_dir="./generated/hidden_states_coco2017_val",
        )

