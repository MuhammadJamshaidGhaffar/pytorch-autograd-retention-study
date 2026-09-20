import os
import platform
import shutil
import subprocess
from datetime import datetime

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return res.stdout.strip() if res.returncode == 0 else "Not installed / Error"
    except Exception:
        return "Not available"

def get_torch_info():
    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda if cuda_avail else "N/A"
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU only"
        return torch_ver, cuda_avail, cuda_ver, device_name
    except ImportError:
        return "Not installed", False, "N/A", "N/A"

def generate_report():
    os.makedirs("docs", exist_ok=True)
    report_path = os.path.join("docs", "environment_report.md")
    
    torch_ver, cuda_avail, cuda_ver, device_name = get_torch_info()
    git_ver = run_cmd("git --version")
    docker_ver = run_cmd("docker --version")
    codeql_ver = run_cmd("codeql version --format=terse")
    
    content = f"""# Environment Report

- **Generated On:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **OS Platform:** {platform.system()} {platform.release()} ({platform.machine()})
- **Python Version:** {platform.python_version()} ({platform.python_implementation()})
- **PyTorch Version:** {torch_ver}
- **CUDA Available:** {cuda_avail}
- **CUDA Runtime Version:** {cuda_ver}
- **Hardware Device:** {device_name}
- **Git Version:** {git_ver}
- **Docker Version:** {docker_ver}
- **CodeQL CLI Version:** {codeql_ver}

## Reproducibility Notes
This environment report was generated to satisfy Module 0 of the study protocol.
All baseline tool executions and benchmark measurements must reference these pinned dependencies.
"""
    with open(report_path, "w") as f:
        f.write(content)
    print(f"[+] Successfully generated: {report_path}")

if __name__ == "__main__":
    generate_report()