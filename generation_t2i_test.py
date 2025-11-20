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
from probes.layer_hooks import register_lasttoken_hooks
import torch
import torch.nn.functional as F



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
target_size_h, target_size_w = 768, 768
patch_size = 16
# image_token_num_per_image = (target_size // patch_size) ** 2  
image_token_num_per_image = 576

q_image_content_conditions = [
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
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    save_name_base: str | None = None,
    pidx: int = 0
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_sample_512', exist_ok=True)
    saved_paths = []
    for i in range(parallel_size):
        name = save_name_base or "img"
        save_path = os.path.join('generated_sample_512', f"{name}_{pidx}.jpg")
        PIL.Image.fromarray(dec[i]).save(save_path)
        saved_paths.append(save_path)
    return saved_paths

@torch.inference_mode()
def generate_with_probe(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    save_name_base: str | None = None,
    pidx: int = 0
):  
    attach_layer_indices(mmgpt.language_model)

    collect_layers_for_E = [0, 5, 10, 20]
    layer_cache, layer_handles = register_lasttoken_hooks(mmgpt.language_model, collect_layers_for_E)

    def reduce_heads(a, method="mean", eps=1e-9):
        if method == "mean":
            return a.mean(dim=0)
        if method == "max":
            return a.max(dim=0).values
        if method == "entropy_weighted":
            p = a / (a.sum(dim=-1, keepdim=True) + eps)
            H = -(p * (p.clamp_min(eps)).log()).sum(dim=-1)
            w = 1.0 / (H + eps)
            w = w / (w.sum() + eps)
            return (a * w[:, None]).sum(dim=0)
        if method == "topk_heads":
            k = min(4, a.size(0))
            score = a.max(dim=-1).values
            vals, idx = score.topk(k)
            w = vals / (vals.sum() + eps)
            return (a[idx] * w[:, None]).sum(dim=0)
        raise ValueError(method)
        
    def _probe_cb(layer_idx, sdpa_probs, probe_ctx):
        cond = sdpa_probs[0::2, :, -1, :]
        a = cond.mean(dim=0).detach()
        vec = reduce_heads(a, method="entropy_weighted")  # [Tk]

        # 存“按层”的序列向量
        probe_ctx.setdefault("seqvec_by_layer", {})[layer_idx] = vec
        
        # nH, Tk = a.shape

        # H, W = probe_ctx["H"], probe_ctx["W"]
        # img_start = probe_ctx["img_start"]
        # img_len = H * W

        # t_img = max(0, min(Tk - img_start, img_len))
        
        # if t_img <= 0:
        #     grid = torch.zeros(H, W, device=a.device, dtype=a.dtype)
        # else:
        #     vec_img = reduce_heads(a[:, img_start: img_start + t_img], method="entropy_weighted")

        #     flat = torch.zeros(img_len, device=a.device, dtype=a.dtype)
        #     flat[:t_img] = vec_img
        #     grid = flat.view(H, W) 
        
        # acc = probe_ctx.setdefault("attn_grid_acc", None)
        # if acc is None:
        #     probe_ctx["attn_grid_acc"] = grid.clone()
        #     probe_ctx["layer_count"] = 1
        # else:
        #     probe_ctx["attn_grid_acc"] += grid
        #     probe_ctx["layer_count"] += 1

        # if t_img > 0:
        #     a_img = a[:, img_start: img_start + t_img]
        #     spars = (a_img**2).sum(dim=-1) / (a_img.sum(dim=-1)**2 + 1e-9)
        #     probe_ctx.setdefault("head_spars_list", []).append(spars.cpu())

    def _probe_cb_triangle(layer_idx, sdpa_probs, probe_ctx):
        if layer_idx != probe_ctx["target_layer"]:
            return

        cond = sdpa_probs[0::2, :, -1, :]        # [B_cond, nH, Tk]
        a = cond.mean(dim=0).detach()            # [nH, Tk]
        Tk = a.size(-1)
        row = Tk - 1

        vec = reduce_heads(a, method="mean")  # [Tk]

        if probe_ctx.get("zero_self", True) and Tk > 0:
            vec[-1] = 0

        # s = float(vec.sum().item())
        # if s > 0:
        #     vec = vec / s

        probe_ctx["A"][row, :Tk] = vec.to(probe_ctx["A"].dtype).cpu()
    
    set_probe_callback(_probe_cb_triangle)

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)
    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    stats = []
    topk_for_mass = 5
    alpha_mass = 0.8
    H, W = img_size // patch_size, img_size  // patch_size
    img_start_pos = 0

    T = image_token_num_per_image
    attn_volume = torch.zeros(T, H, W, dtype=torch.float32, device="cpu")
    L0 = len(input_ids)
    T_img = image_token_num_per_image
    L_total = L0 + T_img
    last_layer = mmgpt.language_model.config.num_hidden_layers - 1
    target_layer = last_layer - 1
    A = torch.zeros((L_total, L_total), dtype=torch.float32)
    A[:] = float("nan")


    past_kv = None
    for i in range(image_token_num_per_image):
        probe_ctx = {
            "A": A,
            "target_layer": target_layer,
            "head_reduce": "entropy_weighted",
            "zero_self": False,
            "H": H, "W": W, "img_start": L0
        }

        layer_cache.clear() 

        with enable_attention_probe(mmgpt.language_model, ctx_payload=probe_ctx):
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_kv,
            )
        past_kv = outputs.past_key_values

        # outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        # 
        probs_cond = torch.softmax(logits, dim=-1)
        p = probs_cond.mean(dim=0, keepdim=True)
        entropy = float(-(p * (p.clamp_min(1e-9)).log()).sum().item())
        top1_p = float(p.max().item())
        topk_mass = float(p.topk(topk_for_mass, dim=-1).values.sum().item())

        # if "attn_grid_acc" in probe_ctx:
        #     grid = probe_ctx["attn_grid_acc"] / max(1, probe_ctx.get("layer_count", 1))
        #     g = (grid - grid.min()) / (grid.max() - grid.min() + 1e-6)
        #     attn_volume[i] = g.cpu()
        # else:
        #     attn_volume[i].zero_()

        # locality_r_by_lh = {}
        # sparsity_by_lh = {}
        # attn_by_layer = probe_ctx.get("attn_by_layer", {})  # {layer_idx: [nH, Tk]}
        
        # cur_pos_1d = probe_ctx["pos_1d"] - 1
        # idx = cur_pos_1d - img_start_pos
        # if 0 <= idx < (H*W):
        #     row = idx // W
        #     col = idx % W
        # else:
        #     row = col = None

        # # 预先构建 key 位置的距离表
        # # Tk 与输入长度相关，每层相同；我们对每层单独计算以避免对齐问题
        # for l, a in attn_by_layer.items():
        #     Tk = a.shape[-1]
        #     dist = torch.empty(Tk, device=a.device)
        #     for kpos in range(Tk):
        #         if row is None:
        #             dist[kpos] = abs(cur_pos_1d - kpos)
        #         else:
        #             r = (kpos - img_start_pos) // W
        #             c = (kpos - img_start_pos) % W
        #             if (kpos < img_start_pos) or (r < 0) or (r >= H):
        #                 # 非图像区：退化成 1D 距离
        #                 dist[kpos] = abs(cur_pos_1d - kpos)
        #             else:
        #                 dist[kpos] = max(abs(row - r), abs(col - c))  # 切比雪夫半径

        #     order = torch.argsort(dist)
        #     for h in range(a.shape[0]):
        #         att = a[h]                          # [Tk]
        #         # C：Herfindahl 稀疏度（越大越“尖”）
        #         sparsity = float((att**2).sum().item())
        #         sparsity_by_lh[(l, h)] = sparsity
        #         # B：locality 半径（累计质量 >= α 的最小半径）
        #         cumsum = torch.cumsum(att[order], dim=0)
        #         idx_r = int((cumsum >= alpha_mass).float().argmax().item())
        #         r_min = float(dist[order[idx_r]].item())
        #         locality_r_by_lh[(l, h)] = r_min

        # layer_entropies = []
        # for l in collect_layers_for_E:
        #     h_last = layer_cache.buf.get(l, None)
        #     if h_last is not None:
        #         logits_l = mmgpt.gen_head(h_last)
        #         p_l = torch.softmax(logits_l, dim=-1)
        #         H_l = float(-(p_l * (p_l.clamp_min(1e-9)).log()).sum().item())
        #         layer_entropies.append(H_l)
        #     else:
        #         layer_entropies.append(float("nan"))

        # stats.append(TokenStat(
        #     step=i,
        #     pos_1d=cur_pos_1d,
        #     row=row, col=col,
        #     entropy=entropy,
        #     top1_p=top1_p,
        #     topk_mass=topk_mass,
        #     layer_entropies=layer_entropies,
        #     locality_r_by_layer_head=locality_r_by_lh or None,
        #     attn_sparsity_by_layer_head=sparsity_by_lh or None,
        # ))

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    saved_paths = []
    for i in range(parallel_size):
        name = save_name_base or "img"
        save_path = os.path.join('generated_samples_with_probe', f"{name}_{i}.jpg")
        PIL.Image.fromarray(dec[i]).save(save_path)
        saved_paths.append(save_path)
    
    print(A.shape)
    probe_home = "probe_runtime"
    os.makedirs(probe_home, exist_ok=True)
    os.makedirs(os.path.join(probe_home, "attention_volume"), exist_ok=True)
    torch.save(A, os.path.join(probe_home, "attention_volume", f"{pidx}_layer{target_layer}.pth"))

    return saved_paths


if __name__ == "__main__":
    gen_kwargs = dict(
        parallel_size=1,
        image_token_num_per_image=image_token_num_per_image,
        img_size=target_size,
        patch_size=patch_size,
    )

    for idx, desc in enumerate(q_image_content_conditions):
        prompt = build_prompt(desc)
        base = slugify_first_words(desc, n=6)

        t0 = time.perf_counter()
        paths = generate(vl_gpt, vl_chat_processor, prompt, save_name_base=f"{idx:02d}-{base}", pidx=idx, img_size=512, patch_size=16, image_token_num_per_image=1024)
        # paths = generate_with_probe(vl_gpt, vl_chat_processor, prompt, save_name_base=f"{idx:02d}-{base}", pidx=idx, **gen_kwargs)
        dt = time.perf_counter() - t0

        print(f"[{idx}] {base}  ->  {dt:.2f}s  | saved: {paths[0]}")

        # torch.cuda.empty_cache()