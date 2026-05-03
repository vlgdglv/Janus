import json, os
from collections import defaultdict
from pathlib import Path
# from cleanfid import fid

DATA_DIR="/home/vlgd/Data/coco2017"

def get_coco2017_prompts(mode="val"):
    ann_path = os.path.join(DATA_DIR, "annotations", f"captions_{mode}2017.json")
    with open(ann_path, "r") as f:
        data = json.load(f)

    id2fname = {im["id"]: im["file_name"] for im in data["images"]}

    caps_by_img = defaultdict(list)
    for ann in data["annotations"]:
        caps_by_img[ann["image_id"]].append(ann["caption"])

    prompts = []
    for image_id, caps in caps_by_img.items():
        fname = id2fname[image_id]
        cap = caps[0]
        prompts.append({"image_id": image_id, "file_name": fname, "caption": cap})

    with open(f"eval_coco/data/coco2017_{mode}_prompts.json", "w") as f:
        json.dump(prompts, f, indent=4)

    print(f"Saved {len(prompts)} prompts to coco2017_{mode}_prompts.json")

def prepare_coco2017_val_fid_stats(
    data_root: str = "~/Data/coco2017",
    stats_name: str = "coco2017_val",
    mode: str = "clean",
):
    root = Path(data_root).expanduser()
    val_dir = root / "val2017"

    if not val_dir.is_dir():
        raise FileNotFoundError(f"val2017 dir not found: {val_dir}")

    print(f"[FID] Preparing custom stats '{stats_name}' from {val_dir} ...")
    fid.make_custom_stats(stats_name, str(val_dir), mode=mode)
    print(f"[FID] Done. You can now use dataset_name='{stats_name}', dataset_split='custom'.")


if __name__ == "__main__":
    # get_coco2017_prompts("val")
    get_coco2017_prompts("train")
    # prepare_coco2017_val_fid_stats()

    pass