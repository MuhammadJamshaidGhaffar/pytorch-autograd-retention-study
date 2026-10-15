"""Record Python, PyTorch, CUDA, GPU, and OS details for Phase 0 provenance."""
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.shared.portable import data as portable_data


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
    output=ROOT/'results/phase_0_recon/runtime_environment.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2)+'\n')
    print('Updated results/phase_0_recon/runtime_environment.json')

if __name__=='__main__':main()
