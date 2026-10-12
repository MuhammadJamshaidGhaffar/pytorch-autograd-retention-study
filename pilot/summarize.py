"""Apply the pilot oracle to preserved evidence; unexpected findings require review."""
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0], lineterminator="\n");w.writeheader();w.writerows(rows)


def read_json(path):
    return json.loads(path.read_text())


def runtime(folder):
    evidence=[]
    for run in range(1,4):
        pair={}
        for v in ['vulnerable','fixed']:
            path=RESULTS/'pilot_runtime'/folder.name/f'{v}_run{run:02}.csv'
            rows=list(csv.DictReader(path.open()))
            assert [int(r['iteration']) for r in rows]==list(range(1,len(rows)+1))
            first,last=rows[0],rows[-1]
            env=read_json(path.with_suffix('.json'))
            pair[v]={'case_id':read_json(folder/'metadata.json')['case_id'],'run_id':run,'variant':v,
                     'iterations':len(rows),'allocated_growth_bytes':int(last['allocated_bytes'])-int(first['allocated_bytes']),
                     'final_graph_nodes':int(last['graph_nodes']),'graph_node_growth':int(last['graph_nodes'])-int(first['graph_nodes']),
                     'final_saved_activation_bytes':int(last['saved_activation_bytes']),
                     'final_payload_bytes':int(last['payload_bytes']),'final_attached_roots':int(last['attached_roots']),
                     'released_by_cleanup_bytes':env['released_by_cleanup_bytes'],'trace':str(path.relative_to(ROOT))}
        v,f=pair['vulnerable'],pair['fixed']
        accepted=(v['graph_node_growth']>0 and v['final_saved_activation_bytes']>0
                  and f['final_graph_nodes']==0 and f['final_saved_activation_bytes']==0
                  and v['allocated_growth_bytes']-f['allocated_growth_bytes']>1024**2
                  and v['released_by_cleanup_bytes']>f['released_by_cleanup_bytes'])
        for row in pair.values():
            row['classification']='CONFIRMED' if accepted and row['variant']=='vulnerable' else ('FIXED_CONTROL' if accepted else 'REVIEW')
            evidence.append(row)
    return evidence


def tool_result(folder,tool,config,variant,output_subdir='pilot_static'):
    base=RESULTS/output_subdir/folder.name/f'{tool}_{config}_{variant}'
    if tool=='codeql':
        execution=RESULTS/output_subdir/f'codeql_{variant}_analyze.execution.json'
        if not execution.exists() or read_json(execution)['status']!='OK':return 'ERROR',[]
        data=read_json(RESULTS/output_subdir/f'codeql_{variant}.sarif')
        findings=[]
        for run in data['runs']:
            for r in run.get('results',[]):
                loc=r['locations'][0]['physicalLocation']
                if loc['artifactLocation']['uri'].startswith(folder.name+'/'):
                    findings.append({'rule':r['ruleId'],'line':loc['region']['startLine'],'message':r['message']['text']})
        return 'OK',findings
    execution=base.with_suffix('.execution.json')
    if not execution.exists() or read_json(execution)['status']!='OK':return 'ERROR',[]
    raw=base.with_suffix('.stdout').read_text()
    if tool=='torch-preflight':
        return 'OK',[{'rule':d['code'],'line':d['line'],'message':d['message']} for d in json.loads(raw)['diagnostics']]
    if tool=='semgrep':
        return 'OK',[{'rule':d['check_id'],'line':d['start']['line'],'message':d['extra']['message']} for d in json.loads(raw)['results']]
    # TorchFix reports diagnostics to stdout; stderr contains its processing summary.
    import re
    findings=[]
    for line in raw.splitlines():
        m=re.search(r':(\d+):\d+: (TOR\d+) (.*)',line)
        if m:findings.append({'rule':m[2],'line':int(m[1]),'message':m[3]})
    if raw.strip() and not findings:return 'REVIEW',[]
    return 'OK',findings


def main():
    folders=sorted(ROOT.glob('cases/case_*'))
    manifest={r['tool_name']:r for r in csv.DictReader((RESULTS/'tools_manifest.csv').open())}
    traces=[];pilot=[];findings=[];pairs=[]
    for folder in folders:
        meta=read_json(folder/'metadata.json')
        evidence=runtime(folder);traces+=evidence
        row={'case_id':meta['case_id'],'valid':'CONFIRMED' if all(e['classification']!='REVIEW' for e in evidence) else 'REVIEW',
             'provenance_kind':meta['provenance_kind'],'independent_real_defect':False}
        for tool,label,configs in [('torch-preflight','torch-preflight',['default','all']),('torchfix','TorchFix',['all']),
                                   ('semgrep','Semgrep',['python']),('codeql','CodeQL',['security-and-quality'])]:
            for config in configs:
                sv,v=tool_result(folder,tool,config,'vulnerable');sf,f=tool_result(folder,tool,config,'fixed')
                # Pre-adjudicated rules seen in this run. Anything new stays REVIEW.
                allowed_lines=set(meta.get('related_sink_lines',[])+meta.get('potential_tensor_source_lines',[]))
                if not allowed_lines:
                    allowed_lines={i+1 for i,line in enumerate((folder/'vulnerable.py').read_text().splitlines()) if '# RETENTION_SINK' in line}
                relevant=[x for x in v if x['rule']=='TG001' and x['line'] in allowed_lines]
                fixed_relevant=[x for x in f if x['rule']=='TG001']
                unknown=[x for x in v if x['rule']=='TG001' and x['line'] not in allowed_lines]+[x for x in v+f if x['rule'] not in {'TG001','py/unused-import'}]
                if sv!='OK' or sf!='OK':verdict='NOT_RUN' if 'NOT_RUN' in (sv,sf) else 'ERROR'
                elif unknown:verdict='REVIEW'
                elif relevant:verdict='NON_SPECIFIC' if fixed_relevant else 'DETECTED'
                else:verdict='MISSED'
                key={'torch-preflight':'torch_preflight','torchfix':'torchfix','semgrep':'semgrep','codeql':'codeql'}[tool]
                row[key+('_all' if config=='all' and tool=='torch-preflight' else '')]=verdict
                pairs.append({'case_id':meta['case_id'],'tool':label,'tool_version':manifest[label]['detected_version'],
                              'configuration':config,'vulnerable_execution':sv,'fixed_execution':sf,'verdict':verdict,
                              'known_to_tool_before_study':meta.get('known_to_tool_before_study','UNKNOWN'),
                              'vulnerable_findings':len(v),'fixed_findings':len(f),
                              'relevant_findings':len(relevant),'present_on_fixed':bool(fixed_relevant)})
                for variant,items in [('vulnerable',v),('fixed',f)]:
                    for item in items:
                        relevant_flag=item['rule']=='TG001' and item['line'] in allowed_lines
                        findings.append({'case_id':meta['case_id'],'tool':label,'tool_version':manifest[label]['detected_version'],
                                         'configuration':config,'variant':variant,'finding_file':str((folder/(variant+'.py')).relative_to(ROOT)),
                                         'finding_line':item['line'],'rule_id':item['rule'],'message':item['message'],
                                         'relevant_to_defect':relevant_flag,
                                         'present_on_fixed':any(x['rule']==item['rule'] for x in f),
                                         'localization_level':'sink_or_flow' if relevant_flag else 'unrelated',
                                         'explanation_level':'retention_and_fix' if relevant_flag else 'unrelated',
                                         'adjudication':'TG001 identifies accumulation on the source-to-sink flow; absent on fixed.' if relevant_flag else 'Unrelated coding-quality diagnostic; not a retention detection.'})
        vmean=statistics.mean(e['allocated_growth_bytes'] for e in evidence if e['variant']=='vulnerable')/2**20
        fmean=statistics.mean(e['allocated_growth_bytes'] for e in evidence if e['variant']=='fixed')/2**20
        row['vulnerable_growth_mib']=round(statistics.mean(e['allocated_growth_bytes'] for e in evidence if e['variant']=='vulnerable')/2**20,4)
        row['fixed_growth_mib']=round(statistics.mean(e['allocated_growth_bytes'] for e in evidence if e['variant']=='fixed')/2**20,4)
        row['excess_graph_growth_mib']=round(vmean-fmean,4)
        pilot.append(row)
    write_csv(RESULTS/'ground_truth.csv',traces)
    write_csv(RESULTS/'pilot_results.csv',pilot)
    write_csv(RESULTS/'baseline_results.csv',pairs)
    if findings:write_csv(RESULTS/'finding_adjudication.csv',findings)
    for row in pilot:print(row['case_id'],row['valid'],row['torch_preflight'],row['excess_graph_growth_mib'])

if __name__=='__main__':main()
