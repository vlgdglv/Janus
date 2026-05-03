"""
Janus-Pro image generation for evaluation datasets — AR / SJD / GSD.

Designed for multi-GPU dispatch via CUDA_VISIBLE_DEVICES:

    step=625
    for i in {0..7}; do
        CUDA_VISIBLE_DEVICES=$i python eval_coco/gen_janus.py \\
            --method gsd --begin $((i*step)) --end $(((i+1)*step)) \\
            --prompt_path /path/to/coco2017_val_prompts.json \\
            --save_dir    /path/to/outputs/gsd_coco \\
            --json_key caption --dataset_name COCO \\
            --do_decode_image \\
            > logs/rank_${i}.log 2>&1 &
    done
    wait
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.models.sjd import (
    DecodeStats,
    _cfg_merge, _embed_cfg, _text_prefill,
    generate_gsd,
    generate_sjd,
    tokens_to_image,
    save_image,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_prompt(caption: str, proc: VLChatProcessor) -> str:
    conv = [{"role": "User", "content": caption}, {"role": "Assistant", "content": ""}]
    s = proc.apply_sft_template_for_multi_turn_prompts(
        conversations=conv, sft_format=proc.sft_format, system_prompt=""
    )
    return s + proc.image_start_tag


@torch.inference_mode()
def _run_ar(
    model, proc, prompt: str,
    temperature: float, cfg_weight: float,
    image_token_num: int, seed: int,
) -> tuple[torch.Tensor, DecodeStats]:
    device    = next(model.language_model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    stats = DecodeStats()
    past, al, ap, _ = _text_prefill(
        model, proc, prompt, cfg_weight, temperature, device, stats
    )
    tokens = torch.zeros(image_token_num, dtype=torch.long, device=device)

    for i in range(image_token_num):
        t         = int(torch.multinomial(ap, 1, generator=generator))
        tokens[i] = t
        emb       = _embed_cfg(model, torch.tensor([t], dtype=torch.long, device=device))
        out       = model.language_model.model(
            inputs_embeds=emb, past_key_values=past, use_cache=True
        )
        past          = out.past_key_values
        stats.total_fwd += 1
        al            = _cfg_merge(model.gen_head(out.last_hidden_state[:, -1, :]), cfg_weight)
        ap            = F.softmax(al / temperature, dim=-1)

    stats.total_tokens = image_token_num
    return tokens, stats


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Janus-Pro evaluation inference — AR / SJD / GSD"
    )

    # dataset
    p.add_argument("--prompt_path",   type=str,  default=None)
    p.add_argument("--dataset_name",  type=str,  default="COCO",
                   choices=["COCO", "laion", "midjourney", "custom"])
    p.add_argument("--split",         type=str,  default="val",
                   choices=["train", "val"])
    p.add_argument("--json_key",      type=str,  default="caption")
    p.add_argument("--id_key",        type=str,  default=None)
    p.add_argument("--begin",         type=int,  default=0)
    p.add_argument("--end",           type=int,  default=None)
    p.add_argument("--name_offset",   type=int,  default=0,
                   help="Added to the global index when naming saved token files.")

    # model
    p.add_argument("--model_path",    type=str,
                   default="/home/vlgd/Models/Janus-Pro-1B/")

    # method
    p.add_argument("--method",        type=str,  default="sjd",
                   choices=["ar", "sjd", "gsd"])
    p.add_argument("--jacobi_window", type=int,  default=8)
    p.add_argument("--max_iter",      type=int,  default=20)
    p.add_argument("--gsd_G",         type=int,  default=16)
    p.add_argument("--gsd_p_thr",     type=float, default=0.15)
    p.add_argument("--gsd_d_thr",     type=float, default=0.5)

    # generation
    p.add_argument("--cfg_guidance_scale", type=float, default=5.0)
    p.add_argument("--temperature",        type=float, default=1.0)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--img_size",           type=int,   default=384)
    p.add_argument("--patch_size",         type=int,   default=16)

    # output
    p.add_argument("--save_dir",        type=str,  default=None)
    p.add_argument("--do_decode_image", action="store_true")
    p.add_argument("--do_save_token",   action="store_true")

    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # resolve prompt path & id key
    default_prompt_paths = {
        "COCO":       f"eval_coco/data/coco2017_{args.split}_prompts.json",
        "laion":      "eval_coco/data/laion_prompts.json",
        "midjourney": "eval_coco/data/midjourney_prompts.json",
    }
    default_id_keys = {
        "COCO": "image_id", "laion": "id", "midjourney": "id", "custom": "id",
    }

    prompt_path = args.prompt_path or default_prompt_paths.get(args.dataset_name)
    if prompt_path is None:
        raise ValueError(
            f"Cannot resolve prompt path for dataset '{args.dataset_name}'. "
            "Pass --prompt_path explicitly."
        )
    id_key = args.id_key or default_id_keys.get(args.dataset_name, "id")

    # save directory
    if args.save_dir is None:
        tag = (f"{args.method}_G{args.gsd_G}" if args.method == "gsd"
               else args.method)
        args.save_dir = os.path.join(
            "generated",
            f"{tag}_{args.dataset_name}_{args.split}_{args.begin}-{args.end or 'end'}"
        )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # load model — always on cuda:0; CUDA_VISIBLE_DEVICES selects the physical GPU
    print(f"Loading model from {args.model_path} ...")
    proc  = VLChatProcessor.from_pretrained(args.model_path)
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).cuda().eval()
    print("Model loaded.")

    # load prompts
    with open(prompt_path) as f:
        all_prompts = json.load(f)

    end    = args.end if args.end is not None else len(all_prompts)
    subset = all_prompts[args.begin : end]
    print(f"Prompts : {prompt_path}  [{args.begin}:{end}]  →  {len(subset)} samples")
    print(f"Method  : {args.method.upper()}"
          + (f"  window={args.jacobi_window}" if args.method != "ar" else ""))
    print(f"Save dir: {args.save_dir}")

    image_token_num = (args.img_size // args.patch_size) ** 2  # 576

    # per-method generation closure
    def run_one(full_prompt: str) -> tuple[torch.Tensor, DecodeStats]:
        if args.method == "ar":
            return _run_ar(
                model, proc, full_prompt,
                temperature=args.temperature,
                cfg_weight=args.cfg_guidance_scale,
                image_token_num=image_token_num,
                seed=args.seed,
            )
        elif args.method == "sjd":
            return generate_sjd(
                model, proc, full_prompt,
                temperature=args.temperature,
                cfg_weight=args.cfg_guidance_scale,
                image_token_num=image_token_num,
                img_size=args.img_size,
                patch_size=args.patch_size,
                jacobi_window=args.jacobi_window,
                max_iter_per_window=args.max_iter,
                seed=args.seed,
            )
        else:  # gsd
            return generate_gsd(
                model, proc, full_prompt,
                temperature=args.temperature,
                cfg_weight=args.cfg_guidance_scale,
                image_token_num=image_token_num,
                img_size=args.img_size,
                patch_size=args.patch_size,
                jacobi_window=args.jacobi_window,
                max_iter_per_window=args.max_iter,
                G=args.gsd_G,
                p_thr=args.gsd_p_thr,
                d_thr=args.gsd_d_thr,
                seed=args.seed,
            )

    # main loop
    per_sample_stats = []   # one dict per image — for JSON summary
    total_time   = 0.0
    agg_fwd      = 0
    agg_accepted = 0
    agg_rejected = 0

    for offset, item in tqdm(enumerate(subset), total=len(subset), desc=args.method.upper()):
        global_idx = args.begin + offset
        caption    = item[args.json_key]
        img_id     = item.get(id_key, global_idx)
        full_prompt = _build_prompt(caption, proc)

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            tokens, stats = run_one(full_prompt)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        total_time   += elapsed
        agg_fwd      += stats.total_fwd
        agg_accepted += stats.n_accepted
        agg_rejected += stats.n_rejected

        per_sample_stats.append({
            "img_id":      img_id,
            "global_idx":  global_idx,
            "latency_s":   round(elapsed, 4),
            "nfe":         stats.total_fwd,
            "n_accepted":  stats.n_accepted,
            "n_rejected":  stats.n_rejected,
        })

        if args.do_decode_image:
            arr = tokens_to_image(model, tokens, args.img_size, args.patch_size)
            save_image(arr, str(save_dir / f"generated_{img_id}.jpg"))

        if args.do_save_token:
            torch.save(
                {"tokens": tokens.cpu()},
                save_dir / f"token_sample_{global_idx + args.name_offset}.pt",
            )

    # aggregate stats
    n  = max(1, len(per_sample_stats))
    ar = agg_accepted / max(1, agg_accepted + agg_rejected)

    summary = {
        "method":          args.method,
        "jacobi_window":   args.jacobi_window if args.method != "ar" else None,
        "gsd_G":           args.gsd_G         if args.method == "gsd" else None,
        "begin":           args.begin,
        "end":             end,
        "n_samples":       n,
        "avg_latency_s":   round(total_time / n, 4),
        "avg_nfe":         round(agg_fwd / n, 2),
        "avg_tok_per_fwd": round(image_token_num / (agg_fwd / n), 3),
        "accept_rate":     round(ar, 4) if args.method != "ar" else None,
        "per_sample":      per_sample_stats,
    }

    stats_path = save_dir / f"stats_{args.begin}-{end}.json"
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'─'*55}")
    print(f"  Method          : {args.method.upper()}"
          + (f"  (window={args.jacobi_window})" if args.method != "ar" else ""))
    print(f"  Samples         : {n}")
    print(f"  Avg latency     : {summary['avg_latency_s']:.3f} s/image")
    print(f"  Avg NFE         : {summary['avg_nfe']:.1f} fwd/image")
    print(f"  Avg tok/fwd     : {summary['avg_tok_per_fwd']:.3f}")
    if args.method != "ar":
        print(f"  Accept rate     : {summary['accept_rate']:.3f}")
    print(f"{'─'*55}")
    print(f"Stats saved to : {stats_path}")
    print(f"Done. {n} samples → {args.save_dir}")


if __name__ == "__main__":
    main()
