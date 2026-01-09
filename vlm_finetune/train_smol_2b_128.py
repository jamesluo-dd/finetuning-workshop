import modal
from pathlib import Path

from .common import checkpoints_volume, data_volume



CONFIG_FILE_PATH = Path("/config_smol_2b_128.yml")

axolotl_image = (
    modal.Image.from_registry("axolotlai/axolotl:main-20260108-py3.11-cu128-2.8.0")
    .pip_install(
        "huggingface_hub",
        "hf-transfer",
        "wandb",
        "fastapi",
        "pydantic",
        "num2words",
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
    .add_local_file(Path(__file__).parent / "config_smol_2b_128.yml", CONFIG_FILE_PATH.as_posix())
)

app = modal.App("axolotl-vlm-finetune_smolvlm_2b")

CKPT_VOLUME_DIR = Path("/checkpoints")
DATA_VOLUME_DIR = Path("/data")

LORA_OUTPUT_DIR = CKPT_VOLUME_DIR / "vlm-lora-out_smolvlm_2b_gptlabel_ZL_v2_128"
MERGED_OUTPUT_DIR = CKPT_VOLUME_DIR / "vlm-merged-out_smolvlm_2b_gptlabel_ZL_v2_128"


@app.function(
    image=axolotl_image,
    gpu="H200",
    #secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={CKPT_VOLUME_DIR.as_posix(): checkpoints_volume, DATA_VOLUME_DIR.as_posix(): data_volume},
    timeout=4 * 60 * 60,  # 4 hours
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

    subprocess.run(
        [
            "axolotl",
            "merge-lora",
            CONFIG_FILE_PATH.as_posix(),
            f"--lora-model-dir={LORA_OUTPUT_DIR.as_posix()}",
            f"--output-dir={MERGED_OUTPUT_DIR.as_posix()}",
        ],
        check=True,
    )
