# -*- coding: utf-8 -*-
"""点选题型单测。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo_captcha.types import click


class TestClick(unittest.TestCase):
    def test_deterministic(self):
        s1 = click.generate(123)
        s2 = click.generate(123)
        import numpy as np
        np.testing.assert_array_equal(s1.captcha.images['full'], s2.captcha.images['full'])
        self.assertEqual(s1.gt, s2.gt)

    def test_easy_solver(self):
        d = click.PRESETS['easy']
        for i in range(8):
            s = click.generate(100 + i, d)
            a = click.ClickSolver().solve(s.captcha)
            m = click.metric(a, s.gt)
            self.assertTrue(m['pass'],
                            f'easy case{i}: {m["fail_reason"]} dev={m.get("dev_mean")}')

    def test_metric_perfect(self):
        s = click.generate(7)
        a = click.Answer(values={'centers': [list(c) for c in s.gt['centers']]})
        m = click.metric(a, s.gt)
        self.assertTrue(m['pass'])
        self.assertAlmostEqual(m['dev_mean'], 0.0)

    def test_metric_missing(self):
        s = click.generate(7)
        a = click.Answer(values={'centers': s.gt['centers'][:-1]})
        m = click.metric(a, s.gt)
        self.assertFalse(m['pass'])
        self.assertEqual(m['fail_reason'], '漏识')


if __name__ == '__main__':
    unittest.main()
