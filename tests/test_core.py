# -*- coding: utf-8 -*-
"""核心框架单测：种子派生、注册表、指标工具、真实样本入口。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo_captcha import core


class TestSeeds(unittest.TestCase):
    def test_derive_seed_stable(self):
        self.assertEqual(core.derive_seed(42, 'slider', 7),
                         core.derive_seed(42, 'slider', 7))

    def test_derive_seed_varies(self):
        seeds = {core.derive_seed(1, 'slider', i) for i in range(100)}
        self.assertEqual(len(seeds), 100)
        self.assertNotEqual(core.derive_seed(1, 'slider', 0), core.derive_seed(2, 'slider', 0))


class TestMetrics(unittest.TestCase):
    def test_wrap180(self):
        self.assertAlmostEqual(core.wrap180(190), -170.0)
        self.assertAlmostEqual(core.wrap180(-190), 170.0)
        self.assertAlmostEqual(core.wrap180(0), 0.0)

    def test_error_stats(self):
        s = core.error_stats([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(s['mean'], 2.5)
        self.assertAlmostEqual(s['median'], 2.5)
        self.assertAlmostEqual(s['max'], 4.0)
        self.assertTrue(np.isnan(core.error_stats([])['mean']))

    def test_jsonable(self):
        out = core.jsonable({'a': np.float64(1.5), 'b': np.int64(2),
                             'c': (1, 2), 'd': np.array([1.0]), 'e': float('nan')})
        self.assertEqual(out, {'a': 1.5, 'b': 2, 'c': [1, 2], 'd': [1.0], 'e': None})


class TestRegistry(unittest.TestCase):
    def test_register_and_get(self):
        spec = core.TypeSpec(name='_t', description='', generate=None, solver=None,
                             metric=None, aggregate=None)
        core.register(spec)
        self.assertIs(core.get_spec('_t'), spec)
        core._REGISTRY.pop('_t')

    def test_duplicate_rejected(self):
        spec = core.TypeSpec(name='_t2', description='', generate=None, solver=None,
                             metric=None, aggregate=None)
        core.register(spec)
        with self.assertRaises(ValueError):
            core.register(spec)
        core._REGISTRY.pop('_t2')


class TestRealSamples(unittest.TestCase):
    def test_load_real_samples(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'slider'
            root.mkdir()
            import cv2
            img = np.zeros((10, 10, 3), np.uint8)
            cv2.imwrite(str(root / 'bg.png'), img)
            (root / 'case1.json').write_text(json.dumps({
                'images': {'bg': 'bg.png'}, 'meta': {}, 'gt': {'x': 3.0}}),
                encoding='utf-8')
            samples = core.load_real_samples(td)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].captcha.type, 'slider')
            self.assertEqual(samples[0].gt['x'], 3.0)


if __name__ == '__main__':
    unittest.main()
