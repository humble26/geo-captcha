# -*- coding: utf-8 -*-
"""空间推理（拼图还原）题型单测。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo_captcha.types import spatial


class TestSpatial(unittest.TestCase):
    def test_deterministic(self):
        s1 = spatial.generate(123)
        s2 = spatial.generate(123)
        import numpy as np
        np.testing.assert_array_equal(s1.captcha.images['full'], s2.captcha.images['full'])
        self.assertEqual(s1.gt, s2.gt)

    def test_gt_is_permutation(self):
        s = spatial.generate(9)
        self.assertEqual(sorted(s.gt['arrangement']), list(range(9)))
        self.assertNotEqual(s.gt['arrangement'], list(range(9)))

    def test_easy_solver(self):
        d = spatial.PRESETS['easy']
        exact = 0
        for i in range(6):
            s = spatial.generate(300 + i, d)
            a = spatial.SpatialSolver().solve(s.captcha)
            m = spatial.metric(a, s.gt)
            exact += m['exact']
        self.assertGreaterEqual(exact, 5, f'easy 6 例仅 {exact} 例还原')

    def test_metric(self):
        s = spatial.generate(7)
        a = spatial.Answer(values={'perm': list(s.gt['arrangement'])})
        m = spatial.metric(a, s.gt)
        self.assertTrue(m['pass'])
        self.assertAlmostEqual(m['tile_acc'], 1.0)

    def test_cost_matrix_broadcast_equivalence(self):
        """广播化代价矩阵与逐对循环实现数值等价（回归防护）。"""
        import numpy as np
        s = spatial.generate(321)
        img = s.captcha.images['full']
        tiles = [img[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100].astype(np.float32)
                 for i in range(3) for j in range(3)]
        Hn, Vn = spatial.SpatialSolver()._cost_matrices(tiles)

        S = spatial.STRIP
        wh = np.linspace(1.8, 0.5, S)[None, :, None]
        wv = np.linspace(1.8, 0.5, S)[:, None, None]
        right = [t[:, -S:, :] for t in tiles]
        left = [t[:, :S, :] for t in tiles]
        bottom = [t[-S:, :, :] for t in tiles]
        top = [t[:S, :, :] for t in tiles]
        var_h = [float(np.mean((r[:, :-1, :] - r[:, 1:, :]) ** 2)) / 255.0 ** 2 for r in right]
        var_v = [float(np.mean((b[:-1, :, :] - b[1:, :, :]) ** 2)) / 255.0 ** 2 for b in bottom]
        H = np.zeros((9, 9))
        V = np.zeros((9, 9))
        for a in range(9):
            for b in range(9):
                if a == b:
                    continue
                h = (np.mean(wh * (right[a] - left[b]) ** 2)
                     + np.mean((2 * right[a][:, -1, :] - right[a][:, -2, :]
                                - left[b][:, 0, :]) ** 2)
                     + np.mean((2 * left[b][:, 0, :] - left[b][:, 1, :]
                                - right[a][:, -1, :]) ** 2)) / 255.0 ** 2
                v = (np.mean(wv * (bottom[a] - top[b]) ** 2)
                     + np.mean((2 * bottom[a][-1, :, :] - bottom[a][-2, :, :]
                                - top[b][0, :, :]) ** 2)
                     + np.mean((2 * top[b][0, :, :] - top[b][1, :, :]
                                - bottom[a][-1, :, :]) ** 2)) / 255.0 ** 2
                H[a, b] = h / (var_h[a] + 1e-4)
                V[a, b] = v / (var_v[a] + 1e-4)
        H[np.diag_indices(9)] = V[np.diag_indices(9)] = 1e9
        # float32 舍入容差；对角线 1e9 在 float32 下精确可表示
        np.testing.assert_allclose(Hn, H, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(Vn, V, rtol=1e-4, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
