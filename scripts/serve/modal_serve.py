"""Deployment definition only. Requires Modal SDK; never launched by offline tests."""
import os
from pathlib import PurePosixPath
import subprocess
import modal

MODEL = os.environ.get('SPRINT_BASE_MODEL', 'Qwen/Qwen3.5-9B')
GPU = os.environ.get('SPRINT_GPU', 'H100:1')
ADAPTER = os.environ.get('SPRINT_ADAPTER_PATH', '/hf-cache/adapters/watchdog')
ADAPTER_NAME = os.environ.get('SPRINT_ADAPTER_NAME', 'watchdog')
TP = int(os.environ.get('SPRINT_TENSOR_PARALLEL', '1'))
MAX_RANK = os.environ.get('SPRINT_MAX_LORA_RANK', '64')
NO_ADAPTER = os.environ.get('SPRINT_NO_ADAPTER', '0')  # '1' serves the base model only (dry run)
if not PurePosixPath(ADAPTER).is_relative_to('/hf-cache') or '..' in PurePosixPath(ADAPTER).parts:
    raise ValueError('Adapter must reside inside the mounted hf-cache volume')

app = modal.App('swarm-sprint-watchdog')
cache = modal.Volume.from_name('hf-cache', create_if_missing=False)
image = (modal.Image.from_registry('nvidia/cuda:12.9.0-devel-ubuntu22.04', add_python='3.12')
         .entrypoint([]).pip_install('vllm==0.32.0', 'transformers==5.19.0')  # 0.21.0 pulled a Transformers without gemma4
         .env({'HF_HOME': '/hf-cache', 'SPRINT_BASE_MODEL': MODEL,
               'SPRINT_ADAPTER_PATH': ADAPTER, 'SPRINT_ADAPTER_NAME': ADAPTER_NAME,
               'SPRINT_TENSOR_PARALLEL': str(TP), 'SPRINT_MAX_LORA_RANK': MAX_RANK, 'SPRINT_NO_ADAPTER': NO_ADAPTER}))


@app.function(image=image, gpu=GPU, volumes={'/hf-cache': cache}, timeout=3600,
              scaledown_window=300, max_containers=1, secrets=[modal.Secret.from_name('sprint-vllm-auth')])
@modal.web_server(port=8000, startup_timeout=1200)
def serve():
    from pathlib import Path
    import json
    base_only = os.environ.get('SPRINT_NO_ADAPTER') == '1'
    adapter = Path(os.environ['SPRINT_ADAPTER_PATH'])
    if not base_only:
        config = json.loads((adapter / 'adapter_config.json').read_text())
        if config.get('base_model_name_or_path') != os.environ['SPRINT_BASE_MODEL']:
            raise ValueError('Adapter base model must match SPRINT_BASE_MODEL')
        if not any((adapter / n).is_file() for n in ('adapter_model.safetensors', 'adapter_model.bin')):
            raise ValueError('Downloaded adapter weights are missing')
    # vLLM reads VLLM_API_KEY from the named Modal secret. Do not pass keys in argv.
    if not os.environ.get('VLLM_API_KEY'):
        raise ValueError('VLLM_API_KEY must be provided by the Modal secret')
    cmd = ['vllm', 'serve', os.environ['SPRINT_BASE_MODEL'], '--host', '0.0.0.0',
           '--port', '8000', '--tensor-parallel-size',
           os.environ['SPRINT_TENSOR_PARALLEL'], '--max-model-len', '8192',
           '--served-model-name', os.environ['SPRINT_BASE_MODEL'], 'base']
    if not base_only:
        cmd += ['--enable-lora', '--lora-modules',
                os.environ['SPRINT_ADAPTER_NAME'] + '=' + str(adapter), '--max-lora-rank',
                os.environ['SPRINT_MAX_LORA_RANK']]
    proc = subprocess.Popen(cmd)

    # If vLLM exits, kill the container at once so a crash never holds a GPU
    # until startup_timeout, and Modal does not keep spawning replacements.
    import threading, time
    def _exit_when_vllm_dies():
        while proc.poll() is None:
            time.sleep(5)
        print(f'vllm exited with code {proc.returncode}; stopping container', flush=True)
        os._exit(1)
    threading.Thread(target=_exit_when_vllm_dies, daemon=True).start()
