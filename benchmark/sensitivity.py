"""Static-only driver sensitivity: remove observer yield, flatten R001 storage alias.

Not additional benchmark cases, and not independent runtime ground truth.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pilot.run_static import execute,DEFAULT_BIN

def main():
    rows=[];out=ROOT/'results/benchmark_sensitivity';out.mkdir(exist_ok=True)
    for folder in sorted((ROOT/'benchmark').glob('R*')):
        for variant in ['vulnerable','fixed']:
            code=(folder/(variant+'.py')).read_text()
            code='\n'.join(line for line in code.split('\n') if not line.lstrip().startswith('yield '))
            if folder.name=='R001' and '    all_scores = c["scores"]' not in code:
                code=code.replace('    model = c["model"]','    model = c["model"]\n    all_scores = c["scores"]')
                code=code.replace("c['scores'][i]",'all_scores[i]')
            path=out/f'{folder.name}_{variant}.py';path.write_text(code)
            rec,raw=execute([DEFAULT_BIN/'torch-preflight','check','--format','json','-j','1',path],out/f'{folder.name}_{variant}',valid_codes=(0,1))
            data=json.loads(raw);assert not data['failures'] and data['summary']['files_checked']==1
            rows.append(dict(case_id=folder.name,variant=variant,status=rec['status'],diagnostics=data['diagnostics']))
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print([(r['case_id'],r['variant'],[d['code'] for d in r['diagnostics']]) for r in rows])
if __name__=='__main__':main()
