# -*- coding: utf-8 -*-
"""Runner 与 CLI 集成单测。"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo_captcha.runner import run_all, write_outputs


class TestRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_all(['slider', 'rotation'], n=2, master_seed=99,
                             out_dir=None)
        cls.tmp = Path(__file__).parent / '_tmp_out'
        cls.report = write_outputs(cls.result, str(cls.tmp))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_blocks(self):
        names = [b['name'] for b in self.result['blocks']]
        self.assertEqual(names, ['slider', 'rotation'])
        for b in self.result['blocks']:
            self.assertEqual(len(b['cases']), 2)
            self.assertIn('pass_rate', b['summary'])

    def test_outputs(self):
        self.assertTrue((self.tmp / 'results.json').exists())
        text = self.report.read_text(encoding='utf-8')
        self.assertIn('# 几何验证码 Benchmark 报告', text)
        self.assertIn('## slider', text)
        data = json.loads((self.tmp / 'results.json').read_text(encoding='utf-8'))
        self.assertEqual(data['master_seed'], 99)

    def test_seed_derivation_visible(self):
        for b in self.result['blocks']:
            for c in b['cases']:
                self.assertIn('seed', c)


if __name__ == '__main__':
    unittest.main()
