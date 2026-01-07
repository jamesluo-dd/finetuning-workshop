import modal
import subprocess
import re
import shutil
import os


from .common import axolotl_image, data_volume, CONFIG_FILE_PATH
from pathlib import Path
import json

DATA_VOLUME_DIR = Path("/data")

app = modal.App("axolotl-vlm-prep-dataset")

@app.function(
    image= modal.Image.debian_slim().pip_install("requests"),
    #gpu="H100",
    volumes={DATA_VOLUME_DIR.as_posix(): data_volume},
)
def prep_dataset():

    src_paths = [
        "vlm_finetune/data/train_merchant_vs_global.jsonl",
        "vlm_finetune/data/val_merchant_vs_global.jsonl",
        "vlm_finetune/data/test_merchant_vs_global.jsonl",
    ]

    dst_paths = [
        DATA_VOLUME_DIR / "mvg_train.jsonl",
        DATA_VOLUME_DIR / "mvg_eval.jsonl",
        DATA_VOLUME_DIR / "mvg_test.jsonl",
    ]

    # def convert_one(src: Path, dst: Path) -> None:
    #     if not src.exists():
    #         return
        

    #     with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    #         for line in fin:
    #             if not line.strip():
    #                 continue
    #             src_obj = json.loads(line)
    #             dst_obj = to_messages(src_obj)
    #             fout.write(json.dumps(dst_obj, ensure_ascii=False) + "\n")
    current_dir = os.getcwd()
    print(current_dir, DATA_VOLUME_DIR)

    for p in range(len(src_paths)):
        shutil.copy(src_paths[p], dst_paths[p])


    subprocess.run(
        [
            "axolotl",
            "preprocess",
            CONFIG_FILE_PATH.as_posix(),
        ],
        check=True,
    )
    