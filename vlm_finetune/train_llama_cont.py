import modal
from pathlib import Path

from .common import checkpoints_volume, data_volume



CONFIG_FILE_PATH = Path("/config_llama_cont.yml")

axolotl_image = (
    modal.Image.from_registry("axolotlai/axolotl:0.12.1")
    .pip_install(
        "huggingface_hub",
        "hf-transfer",
        "wandb",
        "fastapi",
        "pydantic",
    )
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
            AXOLOTL_NCCL_TIMEOUT="60",
        )
    )
    .entrypoint([])
    .add_local_file(Path(__file__).parent / "config_llama_cont.yml", CONFIG_FILE_PATH.as_posix())
)

app = modal.App("axolotl-vlm-finetune_llama3")

CKPT_VOLUME_DIR = Path("/checkpoints")
DATA_VOLUME_DIR = Path("/data")

LORA_OUTPUT_DIR = CKPT_VOLUME_DIR / "vlm-lora-out_llama3_2_vision_ZL_cont"
MERGED_OUTPUT_DIR = CKPT_VOLUME_DIR / "vlm-merged-out_llama3_2_vision_ZL_cont"


@app.function(
    image=axolotl_image,
    gpu="H100",
    #secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CKPT_VOLUME_DIR.as_posix(): checkpoints_volume, DATA_VOLUME_DIR.as_posix(): data_volume},
    timeout=5 * 60 * 60,  # 4 hours
)
def train():
    import subprocess

    subprocess.run(
        [
            "axolotl",
            "train",
            CONFIG_FILE_PATH.as_posix(),
            "--output-dir",
            LORA_OUTPUT_DIR.as_posix(),
        ],
        check=True,
    )

    subprocess.
        [
            "axolotl",
            "merge-lora",
            CONFIG_FILE_PATH.as_posix(),
            f"--lora-model-dir={LORA_OUTPUT_DIR.as_posix()}",
            f"--output-dir={MERGED_OUTPUT_DIR.as_posix()}",
        ],
        check=True,
    )
