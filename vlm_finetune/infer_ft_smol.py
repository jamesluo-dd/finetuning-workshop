import modal

from .common import checkpoints_volume

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        "bitsandbytes==0.45.3",
        "num2words==0.5.13",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})  # faster model transfers
)

MODEL_PATH = "/checkpoints/vlm-merged-out_smolvlm_2b_gptlabel_ZL_v2_cont/merged"  # Local path to Qwen/Qwen2.5-VL-7B-Instruct weights
VLLM_PORT = 8000


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("vllm-smol-inference")

@app.function(
    image=vllm_image,
    gpu="H200",
    #scaledown_window=240, # 4 minutes
    timeout=10 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/checkpoints": checkpoints_volume,
    },
    env={"VLLM_USE_V1": "0"}
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    import subprocess
    import torch
    import os

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")


    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_PATH,
        "--served-model-name",
        MODEL_PATH,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--enforce-eager",
        "--kv-cache-dtype", "fp8", 
        "--max-model-len", "8192" ,
        #"--quantization","fp8",
        #"--gpu-memory-utilization", "0.30",
        "--limit-mm-per-prompt", "image=2",
       #"--quantization", "bitsandbytes", "--load-format", "bitsandbytes"
        #"--load-format", "fp8" 
        # "--tensor-parallel-size", str(N_GPU)
    ]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)