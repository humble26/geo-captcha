# -*- coding: utf-8 -*-
"""题型：空间推理（拼图还原）——3x3 打乱拼图，输出还原排列。

生成器-求解器契约：
  - 300x300 场景（稠密纹理 + 高对比跨界弧线）切成 3x3 个 100x100 拼块，
    按随机排列打乱重排（显示位置 p 放原始拼块 perm[p]，perm 已知但只进 gt）；
  - 答案 A[p] = "位置 p 应放置的拼块"当前的显示位置编号；gt['arrangement'] 与答案同构
    （A[p] = perm.index(p)），逐位相等才算完全还原。

求解器（纯传统算法）：
  - 拼块边界不相似度 = 接缝加权的 SSD + 双向线性预测项（Pomeranz 式：正向预测
    b 的首列、反向预测 a 的末列，消融 60 例 55→58），
    并按拼块自身边界内部变差做归一化（Gallagher MGC 的简化版）——
    归一化是关键：抑制低纹理边缘"看似平滑衔接"的假匹配（消融数据见 README）；
    注意归一化必须只用预测侧自身变差：对称的 sqrt(var_a*var_b) 实测有害（55→32）；
  - 9! = 362880 种排列全枚举、向量化求全局最优（~35ms/例），无搜索质量问题。
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, asdict
from typing import Dict, List

import cv2
import numpy as np

from .. import synth
from ..core import (Answer, Captcha, Sample, Solver, TypeSpec, count_reasons,
                    register)

SIZE = 300
TILE = 100
GRID = 3
STRIP = 12
_WH = np.linspace(1.8, 0.5, STRIP, dtype=np.float32)[None, :, None]   # 越靠近接缝权重越高
_WV = np.linspace(1.8, 0.5, STRIP, dtype=np.float32)[:, None, None]
_PERMS = np.array(list(itertools.permutations(range(GRID * GRID))), dtype=np.int16)


@dataclass
class Difficulty:
    richness: float = 2.2        # 场景纹理密度（低 -> 边界约束弱 -> 更难）
    noise: float = 0.2
    max_fixed: int = 1           # 打乱后允许的"原位拼块"上限


PRESETS = {
    'easy': Difficulty(richness=3.0, noise=0.0, max_fixed=0),
    'normal': Difficulty(),
    'hard': Difficulty(richness=1.2, noise=0.6, max_fixed=3),
}


def generate(seed: int, difficulty: Difficulty = None) -> Sample:
    d = difficulty or Difficulty()
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)

    scene = synth.texture_background(SIZE, SIZE, rng, richness=d.richness,
                                     noise=d.noise)
    # 穿越拼块边界的高对比弧线/直线，提供跨块连续性线索
    for _ in range(7):
        hue = float(rng.integers(0, 360))
        v = int(rng.choice([30, 60, 230, 250]))
        cv2.circle(scene, (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE))),
                   int(rng.integers(50, 150)), synth.hsv_bgr(hue, 70, v),
                   int(rng.integers(2, 6)), lineType=cv2.LINE_AA)
    for _ in range(3):
        cv2.line(scene, (int(rng.integers(0, SIZE)), 0),
                 (int(rng.integers(0, SIZE)), SIZE),
                 synth.hsv_bgr(float(rng.integers(0, 360)), 60, int(rng.choice([40, 240]))),
                 2, lineType=cv2.LINE_AA)

    tiles = [scene[i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE].copy()
             for i in range(GRID) for j in range(GRID)]      # tile k 的原位 = (k//3, k%3)

    while True:
        perm = list(range(GRID * GRID))
        prng.shuffle(perm)
        fixed = sum(1 for p, t in enumerate(perm) if p == t)
        if perm != list(range(GRID * GRID)) and fixed <= d.max_fixed:
            break

    canvas = np.zeros_like(scene)
    for p, t in enumerate(perm):
        r, c = p // GRID, p % GRID
        canvas[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = tiles[t]

    captcha = Captcha(type='spatial', images={'full': canvas},
                      meta={'tile': TILE, 'grid': GRID, 'difficulty': asdict(d)},
                      seed=seed)
    # 还原排列 A[p] = "位置 p 应放的那个拼块"当前的显示位置编号
    arrangement = [perm.index(p) for p in range(GRID * GRID)]
    return Sample(captcha=captcha, gt={'arrangement': arrangement})


class SpatialSolver(Solver):
    """边界代价（SSD+预测，内部变差归一化）+ 9! 全枚举全局最优。"""

    def solve(self, captcha: Captcha) -> Answer:
        tile = captcha.meta['tile']
        grid = captcha.meta['grid']
        img = captcha.images['full']
        tiles = [img[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile].astype(np.float32)
                 for i in range(grid) for j in range(grid)]
        H, V = self._cost_matrices(tiles)
        perm_est, cost = self._exhaustive(H, V, grid)
        return Answer(values={'perm': perm_est}, debug={'cost': cost})

    def _cost_matrices(self, tiles):
        """全对边界代价矩阵（含双向预测项与内部变差归一化）。

        广播实现与逐对循环实现已做数值等价性回归（tests/test_spatial.py）；
        改动任何一处必须同步核对另一处。预测项的下标槽位与语义强绑定：
        正向预测项 = a 的末两列 + b 的首列，反向预测项 = b 的首两列 + a 的末列
        （曾因槽位错位引入过 bug）。
        """
        n = len(tiles)
        s = STRIP

        right = [t[:, -s:, :] for t in tiles]
        left = [t[:, :s, :] for t in tiles]
        bottom = [t[-s:, :, :] for t in tiles]
        top = [t[:s, :, :] for t in tiles]
        # 拼块自身边界内部变差（归一化基准：自身纹理越平滑，越不容许偶然匹配）
        var_h = np.array([float(np.mean((r[:, :-1, :] - r[:, 1:, :]) ** 2)) / 255.0 ** 2
                          for r in right])
        var_v = np.array([float(np.mean((b[:-1, :, :] - b[1:, :, :]) ** 2)) / 255.0 ** 2
                          for b in bottom])

        # 全对代价一次广播算齐：(9, 1, ...) - (1, 9, ...) -> (9, 9, rows, S, 3)
        Rr = np.stack(right)
        Ll = np.stack(left)
        Bb = np.stack(bottom)
        Tt = np.stack(top)
        ssd_h = (_WH * (Rr[:, None] - Ll[None, :]) ** 2).mean(axis=(2, 3, 4))
        ssd_v = (_WV * (Bb[:, None] - Tt[None, :]) ** 2).mean(axis=(2, 3, 4))
        pred_h = ((2 * Rr[:, None, :, -1, :] - Rr[:, None, :, -2, :]
                   - Ll[None, :, :, 0, :]) ** 2).mean(axis=(2, 3))
        pred_b = ((2 * Ll[None, :, :, 0, :] - Ll[None, :, :, 1, :]
                   - Rr[:, None, :, -1, :]) ** 2).mean(axis=(2, 3))
        pred_v = ((2 * Bb[:, None, -1, :, :] - Bb[:, None, -2, :, :]
                   - Tt[None, :, 0, :, :]) ** 2).mean(axis=(2, 3))
        pred_bv = ((2 * Tt[None, :, 0, :, :] - Tt[None, :, 1, :, :]
                    - Bb[:, None, -1, :, :]) ** 2).mean(axis=(2, 3))

        H = (ssd_h + pred_h + pred_b) / (255.0 ** 2 * (var_h[:, None] + 1e-4))
        V = (ssd_v + pred_v + pred_bv) / (255.0 ** 2 * (var_v[:, None] + 1e-4))
        H = H.astype(np.float32)
        V = V.astype(np.float32)
        H[np.diag_indices(n)] = V[np.diag_indices(n)] = np.float32(1e9)
        return H, V

    @staticmethod
    def _exhaustive(H, V, grid) -> tuple:
        """9! 全枚举（行主序），向量化代价，返回 (最优排列, 代价)。"""
        p = _PERMS
        cost = (H[p[:, 0], p[:, 1]] + H[p[:, 1], p[:, 2]] +
                H[p[:, 3], p[:, 4]] + H[p[:, 4], p[:, 5]] +
                H[p[:, 6], p[:, 7]] + H[p[:, 7], p[:, 8]] +
                V[p[:, 0], p[:, 3]] + V[p[:, 1], p[:, 4]] + V[p[:, 2], p[:, 5]] +
                V[p[:, 3], p[:, 6]] + V[p[:, 4], p[:, 7]] + V[p[:, 5], p[:, 8]])
        i = int(np.argmin(cost))
        return _PERMS[i].tolist(), float(cost[i])


def metric(answer: Answer, gt: Dict) -> Dict:
    est = list(answer.values['perm'])
    gt_arr = list(gt['arrangement'])
    tile_acc = sum(1 for a, b in zip(est, gt_arr) if a == b) / len(gt_arr)
    exact = est == gt_arr
    reason = None
    if not exact:
        reason = '局部错位' if tile_acc >= 2 / 3 else '全局错乱'
    return {'pass': exact, 'tile_acc': tile_acc, 'exact': exact,
            'fail_reason': reason}


def aggregate(cases: List[Dict]) -> Dict:
    accs = [c['tile_acc'] for c in cases]
    return {
        'n': len(cases),
        'exact_rate': sum(1 for c in cases if c['exact']) / len(cases),
        'tile_acc_mean': float(np.mean(accs)),
        'tile_acc_min': float(np.min(accs)),
        'fail_reasons': count_reasons(cases),
    }


def visualize(sample: Sample, answer: Answer) -> Dict[str, np.ndarray]:
    vis = sample.captcha.images['full'].copy()
    for p, t in enumerate(answer.values['perm']):
        r, c = p // 3, p % 3
        cv2.putText(vis, str(t), (c * 100 + 4, r * 100 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return {'vis': vis}


register(TypeSpec(
    name='spatial',
    description='空间推理：3x3 拼图还原（输出还原排列）',
    generate=generate,
    solver=SpatialSolver(),
    metric=metric,
    aggregate=aggregate,
    visualize=visualize,
))
