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
from cleanfid import fid

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
    image_paths = [image_dir / p["fil_name"] for p in prompts]

    mean, std = compute_clip_score(image_paths, texts, model_name, pretrained, batch_size, device)
    print("mean: {:.4f}, std: {:.4f}".format(mean, std))


def calc_generated_clip_score(
    model_name: str = "ViT-L-14",
    image_dir: str = "",
    image_name_fmt: str = "",
    pretrained: str = None,
    batch_size: int = 64,
    device: str | None = None,
    N_SAMPLE = 2000,
):
    with open("eval_coco/data/coco2017_val_prompts.json", "r") as f:
        prompts = json.load(f)
    
    prompts = prompts[:N_SAMPLE]
    texts = [p["caption"] for p in prompts]
    image_paths = [os.path.join(image_dir, image_name_fmt.format(p["image_id"])) for p in prompts]

    mean, std = compute_clip_score(image_paths, texts, model_name, pretrained, batch_size, device)
    print("mean: {:.4f}, std: {:.4f}".format(mean, std))


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
    if False:
        score = fid.compute_fid(
            "generated/baseline_coco2017_val",
            dataset_name="coco2017_val",
            dataset_split="custom",
            mode="clean"
        )
        print(score)

    if False:
        score = fid.compute_fid(
            "generated/row_parallel_coco2017_val",
            dataset_name="coco2017_val",
            dataset_split="custom",
            mode="clean"
        )
        print(score)
        calc_generated_clip_score(
            model_name="local-dir:/home/vlgd/Models/vit_large_patch14_clip_224.openai",
            image_dir="generated/row_parallel_coco2017_val",
            image_name_fmt="generated_{}.jpg",
            )
    
    if False:
        score = fid.compute_fid(
            "generated/row_parallel_coco2017_val_1_more_fwd",
            dataset_name="coco2017_val",
            dataset_split="custom",
            mode="clean"
        )
        print(score)
        calc_generated_clip_score(
            model_name="local-dir:/home/vlgd/Models/vit_large_patch14_clip_224.openai",
            image_dir="generated/row_parallel_coco2017_val_1_more_fwd",
            image_name_fmt="generated_{}.jpg",
            )

    if False:
        score = fid.compute_fid(
            "generated/row_parallel_coco2017_val_3_more_fwd",
            dataset_name="coco2017_val",
            dataset_split="custom",
            mode="clean"
        )
        print(score)
        calc_generated_clip_score(
            model_name="local-dir:/home/vlgd/Models/vit_large_patch14_clip_224.openai",
            image_dir="generated/row_parallel_coco2017_val_3_more_fwd",
            image_name_fmt="generated_{}.jpg",
            )
    

    if True:
        score = fid.compute_fid(
            "generated/row_parallel_spec_coco2017_val",
            dataset_name="coco2017_val",
            dataset_split="custom",
            mode="clean"
        )
        print(score)
        calc_generated_clip_score(
            model_name="local-dir:/home/vlgd/Models/vit_large_patch14_clip_224.openai",
            image_dir="generated/row_parallel_spec_coco2017_val",
            image_name_fmt="generated_{}.jpg",
            )

    
