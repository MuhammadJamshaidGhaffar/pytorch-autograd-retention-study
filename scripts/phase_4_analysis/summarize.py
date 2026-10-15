"""Build uniform Phase 4 summaries for cases C001-C009.

This script reads the Phase 2 runtime/emissions traces and Phase 3 scans from
all four pinned static tools. It writes one set of case, ground-truth, baseline,
and gap tables in the same format for every case from C001-C009. C000 is pilot-only and is
summarized separately in Phase 1.
"""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.phase_1_pilot import summarize as helpers

OUT = ROOT / 'results/phase_4_analysis'
TOOLS = [
    ('torch-preflight', 'torch-preflight', ['default', 'all']),
    ('torchfix', 'TorchFix', ['all']),
    ('semgrep', 'Semgrep', ['python']),
    ('codeql', 'CodeQL', ['security-and-quality']),
]
DIMENSIONS = ('flow_scope', 'sink', 'aliasing', 'lifetime', 'accumulation',
              'cleanup', 'autograd_use', 'control_flow')


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    helpers.write_csv(OUT / name, rows)


def main():
    manifests = {row['tool_name']: row for row in csv.DictReader(
        (ROOT / 'results/phase_0_recon/tools_manifest.csv').open())}
    validation_path = OUT / 'numerical_validation.json'
    validation = {row['case_id']: row for row in helpers.read_json(validation_path)} if validation_path.exists() else {}
    hashes = helpers.read_json(ROOT / 'results/phase_3_baseline/static/source_hashes.json')
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

    cases = sorted(path for path in (ROOT / 'cases').glob('case_*') if path.name != 'case_000')
    all_baselines, findings, runtime_truth, case_results, gaps, baseline_metrics = [], [], [], [], [], []
    for folder in cases:
        metadata = helpers.read_json(folder / 'metadata.json')
        case_id = metadata['case_id']
        case_scores = []
        for tool, label, configurations in TOOLS:
            for configuration in configurations:
                status_v, vulnerable = helpers.tool_result(
                    folder, tool, configuration, 'vulnerable', output_subdir='phase_3_baseline')
                status_f, fixed = helpers.tool_result(
                    folder, tool, configuration, 'fixed', output_subdir='phase_3_baseline')
                allowed = set(metadata.get('related_sink_lines', []) +
                              metadata.get('potential_tensor_source_lines', []))
                allowed.update(metadata.get('source_spans', {}).get('vulnerable', []))
                if not allowed:
                    allowed = {line_no for line_no, line in enumerate(
                        (folder / 'vulnerable.py').read_text().splitlines(), 1)
                        if '# RETENTION_SINK' in line}
                relevant = [item for item in vulnerable
                            if item['rule'] == 'TG001' and item['line'] in allowed]
                fixed_relevant = [item for item in fixed if item['rule'] == 'TG001']
                unknown = ([item for item in vulnerable + fixed
                            if item['rule'] not in {'TG001', 'TG008', 'py/unused-import'}] +
                           [item for item in vulnerable
                            if item['rule'] == 'TG001' and item not in relevant])
                verdict = ('ERROR' if status_v != 'OK' or status_f != 'OK' else
                           'REVIEW' if unknown else
                           'NON_SPECIFIC' if relevant and fixed_relevant else
                           'DETECTED' if relevant else 'MISSED')
                score = dict(case_id=case_id, tool=label,
                    tool_version=manifests[label]['detected_version'],
                    configuration=configuration,
                    known_to_tool_before_study=metadata.get('known_to_tool_before_study', 'UNKNOWN'),
                    vulnerable_execution=status_v, fixed_execution=status_f,
                    verdict=verdict, relevant_findings=len(relevant),
                    present_on_fixed=bool(fixed_relevant))
                all_baselines.append(score)
                case_scores.append(score)
                for variant, items in (('vulnerable', vulnerable), ('fixed', fixed)):
                    for item in items:
                        findings.append(dict(case_id=case_id, tool=label,
                            tool_version=manifests[label]['detected_version'],
                            configuration=configuration, variant=variant,
                            finding_file=str((folder / f'{variant}.py').relative_to(ROOT)),
                            finding_line=item['line'], rule_id=item['rule'],
                            message=item['message'],
                            relevant_to_defect=item['rule'] == 'TG001' and item['line'] in allowed,
                            present_on_fixed=any(other['rule'] == item['rule'] for other in fixed),
                            adjudication=('Matches the predeclared defect span.'
                                if item['rule'] == 'TG001' and item['line'] in allowed
                                else 'Unrelated diagnostic; not scored as a retention detection.')))

        paired_runs = []
        for run in range(1, 4):
            pair = {}
            for variant in ('vulnerable', 'fixed'):
                trace = ROOT / 'results/phase_2_runtime' / folder.name / f'{variant}_run{run:02}.csv'
                rows = list(csv.DictReader(trace.open()))
                environment = helpers.read_json(trace.with_suffix('.json'))
                assert [int(row['iteration']) for row in rows] == list(range(1, 121)), trace
                assert environment['device'] == 'cuda' and environment['codecarbon_version'] == '3.3.1'
                first, last = rows[39], rows[-1]
                pair[variant] = dict(case_id=case_id, run_id=run, variant=variant,
                    graph_nodes_at_40=int(first['graph_nodes']),
                    graph_nodes_at_120=int(last['graph_nodes']),
                    graph_growth=int(last['graph_nodes']) - int(first['graph_nodes']),
                    saved_activation_bytes_at_120=int(last['saved_activation_bytes']),
                    allocated_growth_bytes=int(last['allocated_bytes']) - int(first['allocated_bytes']),
                    cleanup_graph_nodes=environment['cleanup_graph_nodes'],
                    trace=str(trace.relative_to(ROOT)))
                pair[variant].update(emissions_kg=environment['emissions_kg'],
                    energy_consumed_kwh=environment['energy_consumed_kwh'],
                    carbon_intensity_gco2e_per_kwh=environment['carbon_intensity_gco2e_per_kwh'],
                    carbon_location_proxy=environment['carbon_location_proxy'])
            accepted = (pair['vulnerable']['graph_growth'] > 0 and
                        pair['fixed']['graph_growth'] == 0 and
                        pair['vulnerable']['cleanup_graph_nodes'] == 0)
            for variant, row in pair.items():
                row['classification'] = ('CONFIRMED' if variant == 'vulnerable' else 'FIXED_CONTROL') if accepted else 'BOUNDARY'
                runtime_truth.append(row)
            paired_runs.append(pair)

        score_by_tool = {(row['tool'], row['configuration']): row for row in case_scores}
        all_tool_score = score_by_tool[('torch-preflight', 'all')]
        case_results.append(dict(case_id=case_id,
            title=metadata.get('title', metadata.get('case_name', folder.name)),
            source_url=metadata.get('source_url', ''),
            mechanism=metadata.get('mechanism', metadata.get('mechanism_cluster', '')),
            full_application_reproduced=metadata.get('full_application_reproduced', False),
            vulnerable_graph_40=paired_runs[0]['vulnerable']['graph_nodes_at_40'],
            vulnerable_graph_120=paired_runs[0]['vulnerable']['graph_nodes_at_120'],
            fixed_graph_120=paired_runs[0]['fixed']['graph_nodes_at_120'],
            median_vulnerable_allocated_growth_bytes=statistics.median(
                pair['vulnerable']['allocated_growth_bytes'] for pair in paired_runs),
            median_fixed_allocated_growth_bytes=statistics.median(
                pair['fixed']['allocated_growth_bytes'] for pair in paired_runs),
            median_vulnerable_emissions_kg=statistics.median(
                pair['vulnerable']['emissions_kg'] for pair in paired_runs),
            median_fixed_emissions_kg=statistics.median(
                pair['fixed']['emissions_kg'] for pair in paired_runs),
            carbon_location_proxy='Montreal, Quebec (assumed)',
            numerical_validation=validation.get(case_id, {}).get('status', 'NOT_RUN'),
            torch_preflight_default=score_by_tool[('torch-preflight', 'default')]['verdict'],
            torch_preflight_all=all_tool_score['verdict'],
            torchfix=score_by_tool[('TorchFix', 'all')]['verdict'],
            semgrep=score_by_tool[('Semgrep', 'python')]['verdict'],
            codeql=score_by_tool[('CodeQL', 'security-and-quality')]['verdict']))

        dimensions = metadata.get('dimensions', {})
        outcomes = {
            'torch_preflight_default': score_by_tool[('torch-preflight', 'default')]['verdict'],
            'torch_preflight_all': all_tool_score['verdict'],
            'torchfix': score_by_tool[('TorchFix', 'all')]['verdict'],
            'semgrep': score_by_tool[('Semgrep', 'python')]['verdict'],
            'codeql': score_by_tool[('CodeQL', 'security-and-quality')]['verdict'],
        }
        outcome_counts = {name: sum(row['verdict'] == name for row in case_scores)
                          for name in ('DETECTED', 'MISSED', 'NON_SPECIFIC', 'REVIEW', 'ERROR')}
        if outcome_counts['DETECTED']:
            gap_status = 'BASELINE_DETECTED'
        elif outcome_counts['NON_SPECIFIC']:
            gap_status = 'NON_SPECIFIC_SIGNAL'
        elif outcome_counts['MISSED'] == len(case_scores):
            gap_status = 'ALL_BASELINES_MISSED'
        else:
            gap_status = 'UNRESOLVED'
        gaps.append(dict(case_id=case_id,
            **{key: dimensions.get(key, 'NOT_RECORDED') for key in DIMENSIONS},
            **outcomes,
            baseline_configurations=len(case_scores),
            configurations_detected=outcome_counts['DETECTED'],
            configurations_missed=outcome_counts['MISSED'],
            configurations_non_specific=outcome_counts['NON_SPECIFIC'],
            gap_status=gap_status,
            evidence_scope='small case reduction', novelty='NOT_ESTABLISHED',
            interpretation=metadata.get('mechanism', metadata.get('mechanism_cluster', ''))))

    assert all(row['verdict'] not in {'ERROR', 'REVIEW'} for row in all_baselines), 'Unresolved static finding'
    for tool, configuration in sorted({(row['tool'], row['configuration']) for row in all_baselines}):
        rows = [row for row in all_baselines
                if row['tool'] == tool and row['configuration'] == configuration]
        # One aggregate row per tool/configuration across all cases.
        baseline_metrics.append(dict(tool=tool, configuration=configuration,
            cases=len({row['case_id'] for row in rows}),
            detected=sum(row['verdict'] == 'DETECTED' for row in rows),
            missed=sum(row['verdict'] == 'MISSED' for row in rows),
            non_specific=sum(row['verdict'] == 'NON_SPECIFIC' for row in rows),
            errors=sum(row['verdict'] == 'ERROR' for row in rows),
            review=sum(row['verdict'] == 'REVIEW' for row in rows),
            fixed_relevant=sum(row['present_on_fixed'] for row in rows)))

    write('runtime_ground_truth.csv', runtime_truth)
    write('case_results.csv', case_results)
    write('baseline_results.csv', all_baselines)
    write('finding_adjudication.csv', findings)
    write('phase4_gap_analysis.csv', gaps)
    write('baseline_metrics.csv', baseline_metrics)
    print(f'cases={len(case_results)} static_pairs={len(all_baselines)} runtime_pairs={len(runtime_truth)}')


if __name__ == '__main__':
    main()
