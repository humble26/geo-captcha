# -*- coding: utf-8 -*-
"""滑块题型单测：生成器确定性、求解器精度、指标。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo_captcha.core import derive_seed
from geo_captcha.types import slider


class TestSlider(unittest.TestCase):
    def test_deterministic(self):
        s1 = slider.generate(123)
        s2 = slider.generate(123)
        for key in s1.captcha.images:
            np.testing.assert_array_equal(s1.captcha.images[key], s2.captcha.images[key])
        self.assertEqual(s1.gt, s2.gt)

    def test_gt_in_range(self):
        for i in range(5):
            s = slider.generate(derive_seed(1, 'slider', i))
            self.assertTrue(100 <= s.gt['x'] <= 237)
            self.assertTrue(15 <= s.gt['y'] <= 87)

    def test_solver_accuracy(self):
        for i in range(12):
            s = slider.generate(derive_seed(2, 'slider', i))
            a = slider.SliderSolver().solve(s.captcha)
            err = abs(a.values['x'] - s.gt['x'])
            self.assertLessEqual(err, slider.TOL_PX,
                                 f'case{i}: err={err:.2f} method={a.debug["method"]}')

    def test_metric_perfect(self):
        s = slider.generate(7)
        a = slider.Answer(values={'x': s.gt['x'], 'y': s.gt['y']})
        m = slider.metric(a, s.gt)
        self.assertTrue(m['pass'])
        self.assertAlmostEqual(m['err_px'], 0.0)

    def test_aggregate(self):
        cases = [{'err_px': 0.1, 'pass': True, 'fail_reason': None, 'method': 'dark_shape',
                  'score': 0.9},
                 {'err_px': 5.0, 'pass': False, 'fail_reason': '定位错误',
                  'method': 'dark_shape', 'score': 0.9}]
        agg = slider.aggregate(cases)
        self.assertAlmostEqual(agg['pass_rate'], 0.5)
        self.assertAlmostEqual(agg['err_max'], 5.0)


if __name__ == '__main__':
    unittest.main()
