"""Portable reporting must preserve measurements and executable command behavior."""
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pilot.portable import text,data,project_frames

class PortableReports(unittest.TestCase):
    def test_project_and_external_paths(self):
        self.assertEqual(text(ROOT/'cases/case_000/vulnerable.py'),'cases/case_000/vulnerable.py')
        external=Path.home()/'external-package/source.py'
        self.assertEqual(text(external),'<home>/external-package/source.py')
        self.assertEqual(text('https://example.org/issues/123'),'https://example.org/issues/123')
    def test_numbers_and_json_remain_valid(self):
        source={'filename':str(ROOT/'pilot/measure.py'),'bytes':36047360,'line':0,'valid':False}
        normalized=json.loads(text(json.dumps(source)))
        self.assertEqual(normalized,data(source))
        self.assertEqual(normalized['bytes'],source['bytes'])
        self.assertIs(normalized['valid'],False)
    def test_native_unknowns_not_invented(self):
        frames=[{'filename':'??','line':0},{'filename':str(ROOT/'cases/case_000/vulnerable.py'),'line':10}]
        self.assertEqual(project_frames(frames),[{'filename':'cases/case_000/vulnerable.py','line':10}])
        self.assertEqual(frames[0]['filename'],'??')
if __name__=='__main__':unittest.main()
