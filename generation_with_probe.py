
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os, time
import PIL.Image

# for probe
import contextlib
from probes.attn_probe_runtime import attach_layer_indices, enable_attention_probe, set_probe_callback, _probe_ctx
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
image_token_num_per_image = 576
H, W = 24, 24

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

def cfg_merge(log_all, cfg_weight=5.0):
    return log_all[0:1] + cfg_weight * (log_all[0:1] - log_all[1:2])


def fire_up_probe_context(model, L_total, target_layer):
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
    
    def _record_attention_score(layer_idx, sdpa_probs, probe_ctx):
        if layer_idx not in probe_ctx["target_layer"]:
            return
        # print(sdpa_probs.shape)
        BS, nH, col_len, row_len = sdpa_probs.shape

        cond = sdpa_probs[0::2, :, :, :]        # [B_cond, col_len, row_len]
        a = cond.mean(dim=0).detach()           # [col_len, row_len]   
        vec = reduce_heads(a, method="mean")    # [col_len, row_len]
        # print(">>>", sdpa_probs.shape)
        cur_row = probe_ctx["cur_row"][layer_idx]
        for col in range(col_len):
            # print(cur_row, col, col_len, vec.shape)
            probe_ctx["attention_scores"][layer_idx][cur_row, :cur_row+1] = vec[col, :cur_row+1].to(probe_ctx["attention_scores"][layer_idx].dtype).cpu()
            cur_row += 1
        probe_ctx["cur_row"][layer_idx] = cur_row
    
    attach_layer_indices(model.language_model)
    set_probe_callback(_record_attention_score)

    cur_rows = {}
    attn_score_dict = {}
    for layers in target_layer: 
        A = torch.zeros((L_total, L_total), dtype=torch.float32)
        A[:] = float("nan")
        attn_score_dict[layers] = A
        cur_rows[layers] = 0

    probe_dict = {
        "target_layer": target_layer,
        "attention_scores": attn_score_dict,
        "cur_row": cur_rows,
    }

    return probe_dict



@torch.inference_mode()
def probe_ar_and_row_parallel(
    mm, proc, prompt, H, W,
    cfg_weight=5.0, temperature=1.0, pidx=0,
    do_sample=True, probe_result_dir=None,
):
    device = next(mm.language_model.parameters()).device
    test_rows = 3
    total_token_length = test_rows * W
    os.makedirs(probe_result_dir, exist_ok=True)

    last_layer = mm.language_model.config.num_hidden_layers - 1
    target_layer = [0, int(last_layer / 3), int(2 * last_layer / 3), last_layer-1]
    output_hidden_states = True

    ########################### AR ###########################
    tok = proc.tokenizer
    ids = torch.LongTensor(tok.encode(prompt)).to("cuda")
    toks = torch.stack([ids, ids]).to("cuda")
    toks[1, 1:-1] = proc.pad_id
    inp = mm.language_model.get_input_embeddings()(toks)
    
    T = H * W
    generated_tokens = torch.zeros((T), dtype=torch.int).cuda()
    
    def renew_hidden_state_dict():
        hs_dict = {}
        for layer in target_layer:
            hs_dict[layer] = []
        return hs_dict

    def extract_hidden_states(output, hs_dict):
        hidden_states = output.hidden_states
        for layer in target_layer:
            hs = hidden_states[layer]
            BS, SEQ_LEN, HIDDEN_DIM = hs.shape
            # print(layer, "->", hs.shape)
            hs = hs[0].detach().cpu() # conditional hidden state
            if SEQ_LEN > 1 and len(hs_dict[layer]) == 0:
                # prefill
                hs = hs[-1:,:]
                SEQ_LEN = 1
            for sq in range(SEQ_LEN):
                hs_dict[layer].append(hs[None, sq])

    def save_hidden_states(hs_dict, name):
        for layer in target_layer:
            print("layer", layer, len(hs_dict[layer]))
            hs = torch.cat(hs_dict[layer], dim=0).detach().cpu()
            print(hs.shape)
            torch.save(hs, os.path.join(probe_result_dir, "{}_hidden_states_{}.pt".format(name, layer)))

    hidden_states_dict = renew_hidden_state_dict()
    probe_context = fire_up_probe_context(mm, L_total=inp.shape[1] + total_token_length, target_layer=target_layer)

    with enable_attention_probe(mm, ctx_payload=probe_context):
        past = None
        for t in range(total_token_length):
            out = mm.language_model.model(inputs_embeds=inp, use_cache=True, past_key_values=past, output_hidden_states=output_hidden_states)
            past = out.past_key_values
            hs = out.last_hidden_state[:, -1, :]

            if output_hidden_states:
                extract_hidden_states(out, hidden_states_dict)
                
            log_all = mm.gen_head(hs)
            log = log_all[1:2] + cfg_weight * (log_all[0:1] - log_all[1:2])

            probs = torch.softmax(log / temperature, dim=-1)        
            if do_sample:
                nxt = torch.multinomial(probs, num_samples=1)
            else:
                nxt = probs.argmax(dim=-1, keepdim=True)
            
            generated_tokens[t] = nxt
            both = torch.cat([nxt, nxt], dim=0).view(-1)
            img_emb = mm.prepare_gen_img_embeds(both)
            inp = img_emb.unsqueeze(1)

    if True:
        save_hidden_states(hidden_states_dict, f"baseline_ar_{pidx}")

    if False:
        for layer_id in target_layer:
            torch.save(probe_context["attention_scores"][layer_id], os.path.join(probe_result_dir, f"baseline_ar_{pidx}_layer{layer_id}.pth"))

    # torch.save(probe_context["attention_scores"], os.path.join(probe_result_dir, f"baseline_ar_{pidx}_layer{probe_context['target_layer']}.pth"))


    ########################### Row Parallel ###########################
    tok = proc.tokenizer
    ids = torch.LongTensor(tok.encode(prompt)).to("cuda")
    toks = torch.stack([ids, ids]).to("cuda")
    toks[1, 1:-1] = proc.pad_id

    inp = mm.language_model.get_input_embeddings()(toks)
    L_prefill = inp.shape[1]

    T = H * W
    generated_tokens = torch.zeros((T), dtype=torch.int64, device="cuda")

    past = None
    prev_row_embs = []
    ar_rows = 1
    hidden_states_dict = renew_hidden_state_dict()
    probe_context = fire_up_probe_context(mm, L_total=L_prefill + total_token_length, target_layer=target_layer)

    with enable_attention_probe(mm, ctx_payload=probe_context):
        for c in range(W * ar_rows):
            out = mm.language_model.model(
                inputs_embeds=inp,
                use_cache=True,
                past_key_values=past,
                output_hidden_states=output_hidden_states
            )

            if output_hidden_states:
                extract_hidden_states(out, hidden_states_dict)

            hs, past = out.last_hidden_state, out.past_key_values
            last_h = hs[:, -1, :]                           # [2, D]
            log = cfg_merge(mm.gen_head(last_h), cfg_weight).squeeze(0) # [V]

            probs = torch.softmax(log / temperature, dim=-1)
            
            if do_sample:
                nxt = torch.multinomial(probs, num_samples=1).squeeze(0) # generator=g
            else:
                nxt = probs.argmax(dim=-1, keepdim=True)

            token_id = int(nxt)
            generated_tokens[0 * W + c] = token_id

            both = torch.tensor([token_id, token_id], device="cuda").view(-1)
            img_emb = mm.prepare_gen_img_embeds(both)       # [2, D]
            prev_row_embs.append(img_emb[0].detach())       # [D]
            inp = img_emb.unsqueeze(1)                      # [2,1,D]

        prev_row_embs = prev_row_embs[-W:]
        prev_row_embs.insert(0, prev_row_embs.pop())
        for r in range(1, test_rows-ar_rows+1):
            pos_base = L_prefill + r * W           
            row_cond = torch.stack(prev_row_embs, dim=0).to(device)          # [W, D]
            step_emb = torch.stack([row_cond, row_cond], dim=0).contiguous() # [2, W, D]
            pos_ids = (torch.arange(W, device=device, dtype=torch.long) + pos_base).unsqueeze(0).expand(2, -1)
            out_prop = mm.language_model.model(
                inputs_embeds=step_emb,
                past_key_values=past,
                position_ids=pos_ids,
                use_cache=True,
                output_hidden_states=output_hidden_states
            )

            if output_hidden_states:
                extract_hidden_states(out_prop, hidden_states_dict)

            log_q = cfg_merge(mm.gen_head(out_prop.last_hidden_state), cfg_weight).squeeze(0)

            q_probs  = F.softmax(log_q / (temperature), dim=-1)   
    
            if do_sample:      # [W, V]
                proposal = torch.multinomial(q_probs, 1).squeeze(-1)      # [W]
            else:
                proposal = q_probs.argmax(dim=-1)

            prev_row_embs = list(mm.prepare_gen_img_embeds(proposal.unsqueeze(0).expand(2, -1))[0].unbind(dim=0))
            final_row = proposal.clone()
            for c in range(W):
                generated_tokens[r * W + c] = int(final_row[c])
            prev_row_embs.insert(0, prev_row_embs.pop())

    if True:
        save_hidden_states(hidden_states_dict, f"row_parallel_{pidx}")

    if False:
        for layer_id in target_layer:
            torch.save(probe_context["attention_scores"][layer_id], os.path.join(probe_result_dir, f"row_parallel_{pidx}_layer{layer_id}.pth"))


    # torch.save(probe_context["attention_scores"], os.path.join(probe_result_dir, f"row_parallel_{pidx}_layer{probe_context['target_layer']}.pth"))

    return None


if __name__ =="__main__":
   
    save_dir = "parallel_test"
    probe_dir = "probe_runtime"
    images = []
    baseline_prob, row_parallel_prob = [], []
    baseline_token, row_parallel_token = [], []

    probe_result_dir = "probe_runtime/probe_ar_and_row_parallel_front3_row_argmax"

    for idx, desc in enumerate(q_image_content_conditions):
        if idx > 4:
            break
            pass
        prompt = build_prompt(desc)
        # base = slugify_first_words(desc, n=6)

        t0 = time.perf_counter()
        probe_ar_and_row_parallel(vl_gpt, vl_chat_processor, prompt, H, W, cfg_weight=3.0, temperature=1.0, 
                                  pidx=idx, do_sample=False, probe_result_dir=probe_result_dir)
        dt = time.perf_counter() - t0

        print(f"[{idx}] ->  {dt:.2f}s  | ")
        
    #     img_size = 384
    #     with torch.no_grad():
    #         dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[1, 8, img_size//patch_size, img_size//patch_size])
    #         dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    #         dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    #         visual_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    #         visual_img[:, :, :] = dec

    #         os.makedirs(save_dir, exist_ok=True)

    #         name = "img"
    #         save_path = os.path.join(save_dir, f"{name}_{idx}.jpg")
    #         PIL.Image.fromarray(dec[0]).save(save_path)
    #         images.append(dec[0])
            
    #         print(f"[{idx}] {base}  ->  {dt:.2f}s  | ") #saved: {paths[0]}")
    
    # if len(images) > 0:
    #     datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    #     row = np.concatenate(images, axis=1)
    #     save_path = os.path.join(save_dir, f"test_{datetime}.jpg")
    #     PIL.Image.fromarray(row).save(save_path)
    