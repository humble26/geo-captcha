# -*- coding: utf-8 -*-
"""旋转题型单测。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo_captcha.core import wrap180
from geo_captcha.types import rotation


class TestRotation(unittest.TestCase):
    def test_deterministic(self):
        s1 = rotation.generate(123)
        s2 = rotation.generate(123)
        np.testing.assert_array_equal(s1.captcha.images['full'], s2.captcha.images['full'])
        self.assertEqual(s1.gt, s2.gt)

    def test_easy_solver(self):
        d = rotation.PRESETS['easy']
        for i in range(8):
            s = rotation.generate(200 + i, d)
            a = rotation.RotationSolver().solve(s.captcha)
            err = abs(wrap180(a.values['theta'] - s.gt['theta']))
            self.assertLessEqual(err, rotation.TOL_DEG, f'easy case{i}: err={err:.2f}')

    def test_metric(self):
        s = rotation.generate(7)
        a = rotation.Answer(values={'theta': s.gt['theta'] + 190})   # 等价 -170
        m = rotation.metric(a, s.gt)
        self.assertAlmostEqual(m['err_deg'], 170.0, places=6)

    def test_fft_cost_curve_matches_direct(self):
        """FFT 互相关与逐档 np.roll 直接计算数值等价（回归防护）。"""
        s = rotation.generate(321)
        solver = rotation.RotationSolver()
        curve = solver._cost_curve(s.captcha.images['full'], s.captcha.meta)
        img, meta = s.captcha.images['full'], s.captcha.meta
        cx, cy = meta['center']
        r_in = meta['r_in']
        n = solver.N_ANG
        angs = np.arange(n) * (360.0 / n)
        rad = np.deg2rad(angs)

        def band(offsets):
            rs = np.asarray([r_in + o for o in offsets], dtype=float)
            rr, aa = np.meshgrid(rs, rad, indexing='ij')
            mx = np.clip((cx + rr * np.cos(aa)).round().astype(int), 0, img.shape[1] - 1)
            my = np.clip((cy + rr * np.sin(aa)).round().astype(int), 0, img.shape[0] - 1)
            return img[my, mx].astype(np.float64).mean(axis=0)

        inner = band(range(-solver.OFF_HI, -solver.OFF_LO + 1))
        outer = band(range(solver.OFF_LO, solver.OFF_HI + 1))
        direct = np.array([np.mean((np.roll(inner, s, axis=0) - outer) ** 2)
                           for s in range(n)])
        self.assertLess(float(np.abs(curve - direct).max()), 1e-6)
        self.assertEqual(int(np.argmin(curve)), int(np.argmin(direct)))


if __name__ == '__main__':
    unittest.main()
