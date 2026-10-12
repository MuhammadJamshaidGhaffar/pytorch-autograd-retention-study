"""Record the current runtime environment without confusing it with baseline tools."""
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone
import torch
from pilot.portable import data as portable_data

ROOT=Path(__file__).resolve().parent


def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=15)
    return p.stdout.strip() if p.returncode==0 else 'UNAVAILABLE: '+p.stderr.strip()


def main():
    gpu=command(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'])
    data={'generated_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
          'os':platform.platform(),'pytorch':torch.__version__,'cuda_runtime':torch.version.cuda,
          'cuda_available':torch.cuda.is_available(),'gpu_driver_memory':gpu,
          'git':command(['git','--version']),'runtime_interpreter':__import__('sys').executable}
    data=portable_data(data)
    (ROOT/'results/runtime_environment.json').write_text(json.dumps(data,indent=2)+'\n')
    text='# Pilot environment\n\n'+''.join(f'- **{k}**: {v}\n' for k,v in data.items())
    text+='''\nRuntime uses system Python with CUDA PyTorch. Static baselines use the adjacent `.venv` and CodeQL CLI. The authoritative baseline versions/configurations are in [tools_manifest.csv](../results/tools_manifest.csv); full Python baseline dependencies are in [requirements.lock](../baselines/requirements.lock). Runtime package inventory: [runtime_requirements.txt](../results/runtime_requirements.txt).\n\nEach runtime trace also has a JSON sidecar recording its seed, warm-up, iteration count, environment, and cleanup measurement. The obsolete historical September report was removed; it is not the environment for these October traces.\n'''
    (ROOT/'docs/environment_report.md').write_text(text)
    print('Updated docs/environment_report.md and results/runtime_environment.json')

if __name__=='__main__':main()
