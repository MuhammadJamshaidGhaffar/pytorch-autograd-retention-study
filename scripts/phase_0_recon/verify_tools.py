"""Verify the authoritative static-tool versions in Phase 0 provenance.

Run with the baseline environment's Python. Normal execution never silently changes
versions. --refresh preserves an archived copy and all reviewed configurations.
"""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def observed_versions():
    found = {}
    for label, package in [('torch-preflight','torch-preflight'),('TorchFix','torchfix'),('Semgrep','semgrep')]:
        try:
            found[label] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found[label] = None
    try:
        p=subprocess.run(['codeql','version','--format=json'],capture_output=True,text=True,check=True,timeout=15)
        found['CodeQL']=json.loads(p.stdout)['version']
    except (OSError,subprocess.SubprocessError,ValueError,KeyError):
        found['CodeQL']=None
    return found


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true');args=parser.parse_args()
    path=ROOT/'results/phase_0_recon/tools_manifest.csv'
    rows=list(csv.DictReader(path.open()))
    found=observed_versions();changes=[];missing=[]
    for row in rows:
        actual=found.get(row['tool_name'])
        if actual is None:missing.append(row['tool_name']);continue
        if actual != row['detected_version']:
            changes.append((row,actual))
    for name in missing:print(f'UNAVAILABLE: {name}; manifest unchanged')
    for row,actual in changes:print(f"MISMATCH: {row['tool_name']}: recorded={row['detected_version']}, installed={actual}")
    if args.refresh and changes and not missing:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        archive=ROOT/'results/phase_0_recon/manifest_history';archive.mkdir(exist_ok=True)
        shutil.copy2(path,archive/f'tools_manifest_{stamp}.csv')
        for row,actual in changes:
            row['notes']+=f" Version changed from {row['detected_version']} to {actual} at {stamp}; rerun affected baselines."
            row['detected_version']=actual
            row['verified_on']=stamp[:8]
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0], lineterminator="\n");w.writeheader();w.writerows(rows)
        print('Updated the Phase 0 tool manifest; earlier evidence retains its version snapshot.')
    elif not changes and not missing:print('All four installed baseline versions match the Phase 0 manifest.')
    if missing or (changes and not args.refresh):raise SystemExit(1)

if __name__=='__main__':main()
