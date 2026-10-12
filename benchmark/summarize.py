"""Rebuild Phase 2–4 tables; fail closed on unreviewed findings or missing evidence."""
import csv,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pilot import summarize as ps

def write(name,rows):ps.write_csv(ROOT/'results'/name,rows)
def main():
    manifests={r['tool_name']:r for r in csv.DictReader((ROOT/'results/tools_manifest.csv').open())}
    validation={r['case_id']:r for r in ps.read_json(ROOT/'results/benchmark_numerical_validation.json')}
    hashes=ps.read_json(ROOT/'results/benchmark_static/source_hashes.json')
    for name,digest in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    def parse(folder,tool,config,variant):
        return ps.tool_result(folder,tool,config,variant,output_subdir='benchmark_static')
    truth=[];pairs=[];findings=[];summary=[];gaps=[]
    for folder in sorted((ROOT/'benchmark').glob('R*')):
        meta=ps.read_json(folder/'metadata.json');cid=folder.name;accepted=True;tracepairs=[]
        for run in range(1,4):
            pair={}
            for variant in ['vulnerable','fixed']:
                path=ROOT/'results/benchmark_runtime'/cid/f'{variant}_run{run:02}.csv'
                rows=list(csv.DictReader(path.open()));env=ps.read_json(path.with_suffix('.json'))
                assert len(rows)==120 and [int(x['iteration']) for x in rows]==list(range(1,121))
                assert env['seed']==41+run and env['device']=='cuda'
                start,end=rows[39],rows[-1]
                pair[variant]=dict(case_id=cid,run_id=run,variant=variant,graph_nodes_at_40=int(start['graph_nodes']),graph_nodes_at_120=int(end['graph_nodes']),graph_growth=int(end['graph_nodes'])-int(start['graph_nodes']),saved_activation_bytes_at_120=int(end['saved_activation_bytes']),allocated_growth_bytes=int(end['allocated_bytes'])-int(start['allocated_bytes']),rss_growth_bytes=int(end['rss_bytes'])-int(start['rss_bytes']),cleanup_graph_nodes=env['cleanup_graph_nodes'],trace=str(path.relative_to(ROOT)))
            good=pair['vulnerable']['graph_growth']>0 and pair['fixed']['graph_growth']==0 and pair['vulnerable']['cleanup_graph_nodes']==0 and validation[cid]['status']=='PASS'
            accepted &= good
            for variant,row in pair.items():row['classification']=('CONFIRMED' if variant=='vulnerable' else 'FIXED_CONTROL') if good else 'BOUNDARY';truth.append(row)
            tracepairs.append(pair)
        classification='CONFIRMED' if accepted else 'BOUNDARY'
        meta['classification']=classification;(folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        for tool,label,configs in [('torch-preflight','torch-preflight',['default','all']),('torchfix','TorchFix',['all']),('semgrep','Semgrep',['python']),('codeql','CodeQL',['security-and-quality'])]:
            for config in configs:
                sv,v=parse(folder,tool,config,'vulnerable');sf,f=parse(folder,tool,config,'fixed')
                relevant=[x for x in v if x['rule']=='TG001' and x['line'] in meta['source_spans']['vulnerable']]
                fixed_relevant=[x for x in f if x['rule']=='TG001']
                unknown=[x for x in v+f if x['rule'] not in {'TG001','TG008','py/unused-import'}]+[x for x in v if x['rule']=='TG001' and x not in relevant]
                verdict='ERROR' if sv!='OK' or sf!='OK' else 'REVIEW' if unknown else 'NON_SPECIFIC' if relevant and fixed_relevant else 'DETECTED' if relevant else 'MISSED'
                pairs.append(dict(case_id=cid,tool=label,tool_version=manifests[label]['detected_version'],configuration=config,ground_truth=classification,known_to_tool_before_study='YES_MECHANISM' if cid in ['R001','R002'] and tool=='torch-preflight' else 'UNKNOWN',vulnerable_execution=sv,fixed_execution=sf,verdict=verdict,relevant_findings=len(relevant),present_on_fixed=bool(fixed_relevant)))
                for variant,items in [('vulnerable',v),('fixed',f)]:
                    for x in items:
                        rel=x['rule']=='TG001'
                        findings.append(dict(case_id=cid,tool=label,tool_version=manifests[label]['detected_version'],configuration=config,variant=variant,known_to_tool_before_study=pairs[-1]['known_to_tool_before_study'],finding_file=str((folder/(variant+'.py')).relative_to(ROOT)),finding_line=x['line'],rule_id=x['rule'],message=x['message'],relevant_to_defect=rel,present_on_fixed=any(z['rule']==x['rule'] for z in f),localization_level='sink_or_flow' if rel else 'unrelated',explanation_level='retention' if rel else 'unrelated',adjudication='Retained graph diagnostic' if rel else 'TG008: seed is set by external runtime harness; no retention explanation. Unused-import diagnostics are unrelated.'))
        summary.append(dict(case_id=cid,classification=classification,source_url=meta['source_url'],mechanism=meta['mechanism'],full_application_reproduced=False,vulnerable_graph_40=tracepairs[0]['vulnerable']['graph_nodes_at_40'],vulnerable_graph_120=tracepairs[0]['vulnerable']['graph_nodes_at_120'],fixed_graph_120=tracepairs[0]['fixed']['graph_nodes_at_120'],median_vulnerable_allocated_growth_bytes=statistics.median(p['vulnerable']['allocated_growth_bytes'] for p in tracepairs),median_fixed_allocated_growth_bytes=statistics.median(p['fixed']['allocated_growth_bytes'] for p in tracepairs),numerical_validation=validation[cid]['status']))
        if accepted and any(p['case_id']==cid and p['tool']=='torch-preflight' and p['configuration']=='all' and p['verdict']=='MISSED' for p in pairs):
            gaps.append(dict(case_id=cid,**meta['dimensions'],evidence_scope='source-grounded reduction',novelty='NOT_ESTABLISHED',interpretation={'R001':'Known storage mechanism; representation sensitivity is not a new contribution.','R002':'Meter update crosses methods; the running total retains prior loss graphs after backward.','R003':'Running mean/covariance updates recursively retain graph history across forwards.','R004':'In-place copying preserves obsolete buffer graph history despite bounded payload.','R005':'Slicing bounds tensor payload but concatenation retains ancestry of discarded elements.'}[cid]))
    write('benchmark_ground_truth.csv',truth);write('benchmark_results.csv',summary);write('benchmark_baseline_results.csv',pairs)
    if findings:write('benchmark_finding_adjudication.csv',findings)
    if gaps:write('benchmark_gap_analysis.csv',gaps)
    # Mechanism grouping deliberately coalesces R002/R003: two reports, one running-statistic family.
    groups={'R001':'stored outputs','R002':'recursive statistics','R003':'recursive statistics','R004':'copy buffer history','R005':'sliding graph window'}
    metrics=[]
    for tool,config in sorted({(p['tool'],p['configuration']) for p in pairs}):
        subset=[p for p in pairs if p['tool']==tool and p['configuration']==config and p['ground_truth']=='CONFIRMED']
        successes=[p for p in subset if p['verdict']=='DETECTED'];denom=len(subset)
        metrics.append(dict(tool=tool,configuration=config,confirmed_source_instances=denom,detected_instances=len(successes),source_instance_recall=len(successes)/denom if denom else '',mechanism_families=len({groups[p['case_id']] for p in subset}),detected_mechanism_families=len({groups[p['case_id']] for p in successes}),localization_accuracy='UNDEFINED_NO_RELEVANT_FINDINGS' if not successes else len(successes)/len(successes),fixed_relevant_pairs=sum(p['present_on_fixed'] for p in subset),errors_or_review=sum(p['verdict'] in ['ERROR','REVIEW'] for p in subset),independent_case_recall='NOT_ESTIMABLE_OVERLAP_UNKNOWN'))
    write('benchmark_metrics.csv',metrics)
    assert all(p['verdict'] not in ['ERROR','REVIEW'] for p in pairs),'Unresolved static findings'
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
