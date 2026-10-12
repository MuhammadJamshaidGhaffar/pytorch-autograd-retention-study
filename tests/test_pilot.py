"""Checks for scientific confounds and failure handling, not a new detector."""
from collections import deque
import importlib.util
from pathlib import Path
import tempfile
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pilot'))
from measure import inspect_state,load,tensors_in
from run_static import execute


class PilotValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_all_fixes_preserve_values_and_remove_graphs(self):
        for folder in sorted((ROOT/'cases').glob('case_*')):
            with self.subTest(case=folder.name):
                results=[]
                for variant in ['vulnerable','fixed']:
                    torch.manual_seed(42)
                    mod=load(folder/(variant+'.py'),variant)
                    work=load(folder/'workload.py','workload')
                    model=mod.make_model('cpu');state={};samples=[]
                    mod.run_loop(model,work.get_workload(5,'cpu'),state,
                                 lambda s:samples.append(inspect_state(s,model)))
                    self.assertEqual(len(samples),5)
                    values=[t.detach().clone() for t in tensors_in(state)]
                    results.append((values,samples))
                self.assertEqual(len(results[0][0]),len(results[1][0]))
                for v,f in zip(results[0][0],results[1][0]):torch.testing.assert_close(v,f)
                self.assertGreater(results[0][1][-1]['graph_nodes'],results[0][1][0]['graph_nodes'])
                self.assertTrue(all(s['graph_nodes']==0 and s['saved_activation_bytes']==0 for s in results[1][1]))

    def test_no_grad_retains_payload_without_autograd(self):
        model=torch.nn.Linear(8,8);history=[]
        with torch.no_grad():
            for _ in range(5):history.append(model(torch.randn(2,8)))
        stats=inspect_state({'history':history},model)
        self.assertEqual(stats['graph_nodes'],0)
        self.assertGreater(stats['payload_bytes'],0)

    def test_overwrite_is_bounded(self):
        model=torch.nn.Linear(8,8);state={};counts=[]
        for _ in range(6):
            state['last']=model(torch.randn(2,8))
            counts.append(inspect_state(state,model)['graph_nodes'])
        self.assertEqual(len(set(counts)),1)

    def test_cleanup_releases_graph(self):
        model=torch.nn.Linear(8,8);state={'history':[]}
        for _ in range(5):
            state['history'].append(model(torch.randn(2,8)))
            state['history'].clear()
            self.assertEqual(inspect_state(state,model)['graph_nodes'],0)

    def test_backward_frees_saved_activations_despite_retained_nodes(self):
        model=torch.nn.Linear(8,8);history=[]
        for _ in range(5):
            output=model(torch.randn(2,8));output.sum().backward();history.append(output)
            model.zero_grad(set_to_none=True)
        stats=inspect_state({'history':history},model)
        self.assertGreater(stats['graph_nodes'],0)
        self.assertEqual(stats['saved_activation_bytes'],0)

    def test_missing_or_failed_tool_is_error(self):
        with tempfile.TemporaryDirectory() as temp:
            record,_=execute(['/definitely/missing/autograd-tool'],Path(temp)/'missing')
            self.assertEqual(record['status'],'ERROR')
            record,_=execute([sys.executable,'-c','raise SystemExit(2)'],Path(temp)/'failed',valid_codes=(0,1))
            self.assertEqual(record['status'],'ERROR')

if __name__=='__main__':unittest.main()
