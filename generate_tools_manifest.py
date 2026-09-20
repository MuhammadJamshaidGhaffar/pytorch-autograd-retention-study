import csv
import importlib.metadata
import os
import subprocess

def get_pkg_version(pkg_name):
    """Retrieve installed version from python environment."""
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return "Not installed"

def get_cli_version(cmd):
    """Retrieve CLI tool version string."""
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            return res.stdout.strip().split("\n")[0]
        return "Not installed in PATH"
    except Exception:
        return "Not available"

def generate_manifest():
    os.makedirs("results", exist_ok=True)
    manifest_path = os.path.join("results", "tools_manifest.csv")
    
    # 1. Query installed tool versions dynamically
    tp_version = get_pkg_version("torch-preflight")
    tf_version = get_pkg_version("torchfix")
    semgrep_version = get_pkg_version("semgrep")
    if semgrep_version == "Not installed":
        semgrep_version = get_cli_version("semgrep --version")
    codeql_version = get_cli_version("codeql version --format=terse")
    sonar_version = get_cli_version("sonar-scanner --version")

    # 2. Define protocol-mandated metadata
    rows = [
        {
            "tool_name": "torch-preflight",
            "detected_version": tp_version,
            "role": "Principal Direct Baseline",
            "repo_url": "https://github.com/highwaterlabs/torch-preflight",
            "license": "Apache-2.0 / MIT",
            "default_rules": "13 static rules + VRAM guard",
            "status": "Active (Released Aug 2026)",
            "notes": "Direct competitor; explicitly detects retained autograd graphs (e.g. append to outer list), loops, and detach/no_grad."
        },
        {
            "tool_name": "TorchFix",
            "detected_version": tf_version,
            "role": "PyTorch Linter Baseline",
            "repo_url": "https://github.com/pytorch-labs/torchfix",
            "license": "BSD-3-Clause",
            "default_rules": "TOR0XX, TOR1XX, TOR2XX enabled; TOR4XX disabled by default",
            "status": "Archived (Jan 28 2026)",
            "notes": "LibCST-based linter for PyTorch API deprecations; not primarily an autograd retention analyzer. Must run with all rules enabled."
        },
        {
            "tool_name": "CodeQL",
            "detected_version": codeql_version,
            "role": "Generic Static Baseline & L0-L3 Engine",
            "repo_url": "https://github.com/github/codeql",
            "license": "MIT / GitHub Terms",
            "default_rules": "python-queries / default-suite",
            "status": "Active",
            "notes": "Generic taint/data-flow baseline out-of-the-box. Also serves as the query framework for implementing proposed L0-L3 detectors."
        },
        {
            "tool_name": "Semgrep",
            "detected_version": semgrep_version,
            "role": "Generic Rule/Pattern Baseline",
            "repo_url": "https://github.com/semgrep/semgrep",
            "license": "LGPL-2.1",
            "default_rules": "p/default, p/python",
            "status": "Active",
            "notes": "Pattern-matching and local data-flow baseline; does not model dynamic PyTorch autograd graph lifetimes out-of-the-box."
        },
        {
            "tool_name": "SonarQube",
            "detected_version": sonar_version,
            "role": "Industrial Static Baseline",
            "repo_url": "https://github.com/SonarSource/sonarqube",
            "license": "LGPL-3.0",
            "default_rules": "Sonar Way (Python)",
            "status": "Active",
            "notes": "Generic bug/vulnerability analyzer; lacks framework-specific tensor graph reachability modeling."
        }
    ]

    fieldnames = [
        "tool_name", "detected_version", "role", "repo_url",
        "license", "default_rules", "status", "notes"
    ]

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Successfully generated: {manifest_path}")

if __name__ == "__main__":
    generate_manifest()