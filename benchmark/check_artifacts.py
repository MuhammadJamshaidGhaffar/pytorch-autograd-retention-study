"""Validate the review package without rerunning measurements or modifying evidence."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def check():
    counts={};errors=[]
    for cohort,steps,cases in [('pilot',80,7),('benchmark',120,5)]:
        static=ROOT/'results'/f'{cohort}_static'
        assert (static/'tools_manifest.csv').read_bytes()==(ROOT/'results/tools_manifest.csv').read_bytes()
        hashes=json.loads((static/'source_hashes.json').read_text());assert len(hashes)==2*cases
        for path,digest in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
        execution=list(static.rglob('*.execution.json'))
        for p in execution:
            data=json.loads(p.read_text())
            if data['status']!='OK':errors.append(str(p.relative_to(ROOT)))
        traces=list((ROOT/'results'/f'{cohort}_runtime').glob('*/*.csv'));assert len(traces)==6*cases
        for p in traces:
            rows=list(csv.DictReader(p.open()));assert [int(r['iteration']) for r in rows]==list(range(1,steps+1)),str(p)
            env=json.loads(p.with_suffix('.json').read_text());assert env['torch']=='2.14.0+cu130' and env['device']=='cuda'
        counts[cohort]={'cases':cases,'runtime_processes':len(traces),'steps':steps,'static_executions':len(execution)}
    assert not errors,errors
    assert not (ROOT/'supplementary').exists()
    for name in ['pdf_batch','pdf_requests']:
        assert not list((ROOT/'results').glob('pilot_*/'+name))
    protocol=(ROOT/'docs/phase2_protocol.md').read_bytes()
    assert hashlib.sha256(protocol).hexdigest()==json.loads((ROOT/'results/phase2_protocol_freeze.json').read_text())['sha256']
    for table,expected in [('baseline_results.csv',35),('benchmark_baseline_results.csv',25)]:
        rows=list(csv.DictReader((ROOT/'results'/table).open()));assert len(rows)==expected
        assert all(r['verdict'] not in ['ERROR','REVIEW','NOT_RUN'] for r in rows)
    semantic=json.loads((ROOT/'results/benchmark_numerical_validation.json').read_text());assert len(semantic)==5 and all(x['status']=='PASS' for x in semantic)
    return dict(cohorts=counts,total_runtime_processes=72,paired_baseline_configuration_rows=60,baseline_execution_errors=errors,source_hashes_match=True,protocol_hash_matches=True,benchmark_numerical_pairs_passed=5,pdf_experiments_removed=True,phase5_started=False)
if __name__=='__main__':
    result=check();print(json.dumps(result,indent=2));(ROOT/'results/artifact_validation.json').write_text(json.dumps(result,indent=2)+'\n')
