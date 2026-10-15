"""Run the pinned static-analysis baselines for pilot and study cases.

Each tool scans the vulnerable and fixed source files without changing them.
The output preserves raw stdout/stderr, execution status, source hashes, and a
copy of the exact tool manifest. This script gathers findings but does not
adjudicate them; the summary scripts handle that separate analysis step.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.shared.portable import text as portable_text, data as portable_data
DEFAULT_BIN=ROOT.parent/'.venv/bin'
OUT=ROOT/'results/phase_1_pilot/static'


def execute(command, destination, cwd=ROOT, valid_codes=(0,), timeout=300):
    destination.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    try:
        p=subprocess.run(list(map(str,command)),cwd=cwd,text=True,capture_output=True,timeout=timeout,
                         env={**os.environ,'SEMGREP_SEND_METRICS':'off'})
        stdout,stderr,code=p.stdout,p.stderr,p.returncode
        status='OK' if code in valid_codes else 'ERROR'
    except (OSError,subprocess.TimeoutExpired) as e:
        stdout,stderr,code='',str(e),None
        status='ERROR'
    stdout,stderr=portable_text(stdout),portable_text(stderr)
    destination.with_suffix('.stdout').write_text(stdout)
    destination.with_suffix('.stderr').write_text(stderr)
    record={'command':list(map(str,command)),'cwd':str(cwd),'returncode':code,'status':status,
            'seconds':time.perf_counter()-start}
    destination.with_suffix('.execution.json').write_text(json.dumps(portable_data(record),indent=2)+'\n')
    return record,stdout


def main(argv=None, default_phase='pilot'):
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['pilot','study'],default=default_phase,help='pilot scans C000-C005; study scans C001-C009')
    p.add_argument('--tools',nargs='+',default=['torch-preflight','torchfix','semgrep','codeql'])
    p.add_argument('--bin-dir',type=Path,default=DEFAULT_BIN)
    a=p.parse_args(argv)
    global OUT
    OUT=ROOT/({"pilot":"results/phase_1_pilot/static","study":"results/phase_3_baseline/static"}[a.phase])
    a.bin_dir=a.bin_dir.resolve()
    manifest={r['tool_name']:r for r in csv.DictReader((ROOT/'results/phase_0_recon/tools_manifest.csv').open())}
    # Never silently compare a different installed Python baseline to the pinned manifest.
    for tool,label in [('torch-preflight','torch-preflight'),('torchfix','TorchFix'),('semgrep','Semgrep')]:
        if tool in a.tools:
            version=subprocess.check_output([a.bin_dir/'python','-c',
                f'import importlib.metadata; print(importlib.metadata.version({tool!r}))'],text=True).strip()
            if version != manifest[label]['detected_version']:
                raise RuntimeError(f'{tool}: installed {version} differs from results/phase_0_recon/tools_manifest.csv')
    if 'codeql' in a.tools:
        actual=json.loads(subprocess.check_output(['codeql','version','--format=json'],text=True))['version']
        if actual != manifest['CodeQL']['detected_version']:
            raise RuntimeError('CodeQL version differs from results/phase_0_recon/tools_manifest.csv')
    folders=sorted(ROOT.glob('cases/case_*'))
    folders=[d for d in folders if d.name <= 'case_005'] if a.phase == 'pilot' else [d for d in folders if d.name != 'case_000']
    source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                   for d in folders for f in [d/'vulnerable.py',d/'fixed.py']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'source_hashes.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    (OUT/'tools_manifest.csv').write_bytes((ROOT/'results/phase_0_recon/tools_manifest.csv').read_bytes())
    all_rules='TG001,TG002,TG003,TG004,TG005,TG006,TG007,TG008,TG010,TG011,TG012,TG013,TG014'
    for d in folders:
        for variant in ['vulnerable','fixed']:
            f=d/(variant+'.py')
            for tool in a.tools:
                if tool=='codeql':continue
                configurations=['default','all'] if tool=='torch-preflight' else ['all' if tool=='torchfix' else 'python']
                for config in configurations:
                    dest=OUT/d.name/f'{tool}_{config}_{variant}'
                    if tool=='torch-preflight':
                        command=[a.bin_dir/tool,'check','--format','json','-j','1']
                        if config=='all':command+=['--select',all_rules]
                        command+=[f]
                        valid=(0,1)
                    elif tool=='torchfix':
                        command=[a.bin_dir/tool,'--select','ALL','-j','1',f];valid=(0,1)
                    elif tool=='semgrep':
                        command=[a.bin_dir/tool,'scan','--config',ROOT/'baselines/semgrep-python.yaml',
                                 '--json','--metrics','off','--disable-version-check','--no-git-ignore',f];valid=(0,)
                    else:
                        raise ValueError(tool)
                    record,stdout=execute(command,dest,valid_codes=valid)
                    if record['status']=='OK' and tool in ['torch-preflight','semgrep']:
                        try:
                            data=json.loads(stdout)
                            failures=data.get('failures',[]) if tool=='torch-preflight' else data.get('errors',[])
                            if failures:raise ValueError(str(failures))
                            if tool=='torch-preflight' and data['summary']['files_checked']!=1:
                                raise ValueError('Expected one scanned file')
                            if tool=='semgrep' and len(data.get('paths',{}).get('scanned',[]))!=1:
                                raise ValueError('Expected one scanned file')
                        except (ValueError,KeyError) as e:
                            record.update(status='ERROR',validation_error=str(e))
                            dest.with_suffix('.execution.json').write_text(json.dumps(portable_data(record),indent=2)+'\n')
                    print(d.name,variant,tool,config,record['status'],flush=True)
    if 'codeql' in a.tools:
        for variant in ['vulnerable','fixed']:
            with tempfile.TemporaryDirectory(prefix='autograd-codeql-') as temp:
                temp=Path(temp);src=temp/'src';src.mkdir()
                for d in folders:
                    target=src/d.name;target.mkdir()
                    shutil.copy2(d/(variant+'.py'),target/(variant+'.py'))
                dest=OUT/f'codeql_{variant}_create'
                record,_=execute(['codeql','database','create',temp/'db','--language=python',
                                  '--source-root',src,'--threads=2'],dest,timeout=600)
                if record['status']!='OK':continue
                execute(['codeql','database','analyze',temp/'db',
                         'codeql/python-queries@1.8.10:codeql-suites/python-security-and-quality.qls',
                         '--format=sarif-latest','--output',OUT/f'codeql_{variant}.sarif',
                         '--threads=2','--ram=4096'],OUT/f'codeql_{variant}_analyze',timeout=1200)
                sarif=OUT/f'codeql_{variant}.sarif'
                if sarif.exists():sarif.write_text(portable_text(sarif.read_text()))
                print('codeql',variant,'finished',flush=True)

if __name__=='__main__':main()
