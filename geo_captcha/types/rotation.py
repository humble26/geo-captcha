# -*- coding: utf-8 -*-
"""题型：旋转——估计内盘相对正确位置的偏转角（需回转的角度取相反方向）。

生成器-求解器契约：
  - 360x360 图，中心 (180,180)；外环 [r_in+seam, r_out] 取自原始场景，
    内盘 [0, r_in] 取自同一场景旋转 theta 后的内容，两者纹理同源；
  - 求解器把图做极坐标展开，内带相对外带做循环移位搜索，
    答案 = 使内外带连续的内盘偏转角 theta（度，(-180, 180]）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List

import cv2
import numpy as np

from .. import synth
from ..core import (Answer, Captcha, Sample, Solver, TypeSpec, count_reasons,
                    error_stats, register, wrap180)

SIZE = 360
CX = CY = SIZE // 2
R_IN, R_OUT, SEAM = 128, 172, 3
TOL_DEG = 3.0


@dataclass
class Difficulty:
    angle_range: float = 170.0        # |theta| 的上限
    min_angle: float = 8.0            # |theta| 的下限（太接近 0 无意义）
    richness: float = 1.6             # 场景纹理丰富度（低 -> 更难）
    noise: float = 0.3
    spokes: bool = True               # 辐条：跨接缝的长结构，是对齐信号的主要载体


PRESETS = {
    'easy': Difficulty(angle_range=60.0, min_angle=15.0, noise=0.0, richness=2.0),
    'normal': Difficulty(),
    'hard': Difficulty(richness=0.5, noise=0.8),
}


def _scene(rng: np.random.Generator, d: Difficulty) -> np.ndarray:
    """旋转信息量充足的场景：底纹 + 离心斑块（+ 辐条，仅 hard：刻意制造周期歧义）。"""
    img = synth.texture_background(SIZE, SIZE, rng, richness=1.6 * d.richness, noise=d.noise)
    hue = float(rng.integers(0, 360))
    if d.spokes:
        for k in range(7):
            ang = 2 * np.pi * k / 7 + float(rng.uniform(-0.2, 0.2))
            x2 = int(CX + (R_OUT - 6) * math.cos(ang))
            y2 = int(CY + (R_OUT - 6) * math.sin(ang))
            cv2.line(img, (CX, CY), (x2, y2),
                     synth.hsv_bgr((hue + k * 40) % 360, 55, 60), 2, lineType=cv2.LINE_AA)
    # 离心斑块：打破角度均匀性（辐条模式下不足以完全消除 51.4° 倍角歧义谷，
    # 这正是 hard 档的失败模式来源，报告中如实呈现）
    for _ in range(14):
        ang = float(rng.uniform(0, 2 * np.pi))
        r = float(rng.uniform(R_IN * 0.2, R_OUT * 0.9))
        cv2.circle(img, (int(CX + r * math.cos(ang)), int(CY + r * math.sin(ang))),
                   int(rng.integers(10, 26)),
                   synth.hsv_bgr((hue + rng.integers(0, 360)) % 360, 60,
                                 int(rng.integers(40, 230))), -1, lineType=cv2.LINE_AA)
    return img


def generate(seed: int, difficulty: Difficulty = None) -> Sample:
    d = difficulty or Difficulty()
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(d.min_angle, d.angle_range) *
                  (1 if rng.random() < 0.5 else -1))

    base = _scene(rng, d)
    full = base.copy()
    # 内盘 = 原场景旋转 theta 后的圆盘内容
    rot = cv2.getRotationMatrix2D((CX, CY), theta, 1.0)
    rotated = cv2.warpAffine(base, rot, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    disc = ((xx - CX) ** 2 + (yy - CY) ** 2) <= (R_IN - 1) ** 2
    full[disc] = rotated[disc]
    # 接缝环：视觉上分离内外
    cv2.circle(full, (CX, CY), R_IN, (28, 28, 28), SEAM, lineType=cv2.LINE_AA)

    captcha = Captcha(type='rotation', images={'full': full},
                      meta={'center': [CX, CY], 'r_in': R_IN, 'r_out': R_OUT,
                            'seam': SEAM, 'difficulty': asdict(d)},
                      seed=seed)
    return Sample(captcha=captcha, gt={'theta': theta})


class RotationSolver(Solver):
    """极坐标展开 + 内外带循环移位匹配，抛物线细化到 0.05°。

    采样带紧贴接缝（两侧各偏移 OFF_LO..OFF_HI，径向间隔仅 ~4px）：
    消融实验（60 官方种子）显示间隔 12px 时 MAE 11.3°、最大 162.7°（随机斑块
    场景在两个相距较远的圆环上内容相关性弱），收到 4px 后 MAE 0.10°、最大 0.47°
    ——跨接缝的内容连续性是唯一可靠的对齐信号，带距必须小。
    """

    N_ANG = 720          # 角度采样数（0.5°/格）
    OFF_LO = 2           # 采样带相对 r_in 的最近偏移（接缝 3px 时恰在缝缘外）
    OFF_HI = 8           # 最远偏移

    def solve(self, captcha: Captcha) -> Answer:
        n = self.N_ANG
        costs = self._cost_curve(captcha.images['full'], captcha.meta)

        best = int(np.argmin(costs))
        # 抛物线亚格细化（0.5°/格）
        l, c, r = costs[(best - 1) % n], costs[best], costs[(best + 1) % n]
        den = l - 2 * c + r
        frac = 0.0
        if den > 1e-9:
            frac = max(-0.5, min(0.5, 0.5 * (l - r) / den))
        est = wrap180((best + frac) * (360.0 / n))

        # 歧义归因：次优谷（偏离最优 >=15°）与最优之比
        away = np.ones(n, bool)
        for off in range(int(15 / (360.0 / n))):
            away[(best + off) % n] = False
            away[(best - off) % n] = False
        second = float(costs[away].min())
        margin = (second - costs[best]) / max(costs[best], 1e-9)
        return Answer(values={'theta': est},
                      debug={'margin': margin, 'cost': float(costs[best])})

    def _cost_curve(self, img, meta) -> np.ndarray:
        """内/外带循环移位 SSD 代价曲线（长度 N_ANG）。

        FFT 互相关实现与逐档 np.roll 直接计算已做数值等价性回归
        （tests/test_rotation.py，最大差 ~1e-11）；改动须同步核对。
        """
        cx, cy = meta['center']
        r_in = meta['r_in']
        n = self.N_ANG
        angs = np.arange(n) * (360.0 / n)
        rad = np.deg2rad(angs)

        def band(offsets):
            rs = np.asarray([r_in + o for o in offsets], dtype=float)
            rr, aa = np.meshgrid(rs, rad, indexing='ij')
            mx = np.clip((cx + rr * np.cos(aa)).round().astype(int), 0, img.shape[1] - 1)
            my = np.clip((cy + rr * np.sin(aa)).round().astype(int), 0, img.shape[0] - 1)
            return img[my, mx].astype(np.float64).mean(axis=0)  # (n_ang, 3)

        inner = band(range(-self.OFF_HI, -self.OFF_LO + 1))
        outer = band(range(self.OFF_LO, self.OFF_HI + 1))

        # SSD(s) = mean((roll(inner,s) - outer)^2) = A + B - 2*C(s)，
        # 其中 C(s) 是内/外带沿角度轴的循环互相关——用 FFT 一次算齐全部 720 档，
        # 替代逐档 np.roll 的 Python 循环（~4x）。
        A = float((inner ** 2).mean())
        B = float((outer ** 2).mean())
        Fi = np.fft.rfft(inner, axis=0)                   # inner/outer 已按半径取均值：(n, 3)
        Fo = np.fft.rfft(outer, axis=0)
        C = np.fft.irfft(np.conj(Fi) * Fo, n, axis=0)     # (n, 3)
        return A + B - 2.0 * C.sum(axis=1) / C.size


def metric(answer: Answer, gt: Dict) -> Dict:
    err = abs(wrap180(float(answer.values['theta']) - float(gt['theta'])))
    reason = None
    if err > TOL_DEG:
        reason = '歧义图案' if answer.debug.get('margin', 1.0) < 0.15 else '估计偏差'
    return {'pass': err <= TOL_DEG, 'err_deg': err, 'fail_reason': reason,
            'margin': answer.debug.get('margin')}


def aggregate(cases: List[Dict]) -> Dict:
    errs = [c['err_deg'] for c in cases]
    s = error_stats(errs)
    return {
        'n': len(cases),
        'pass_rate': sum(1 for c in cases if c['pass']) / len(cases),
        'mae': s['mean'], 'err_median': s['median'],
        'err_p95': s['p95'], 'err_max': s['max'],
        'tol_deg': TOL_DEG,
        'fail_reasons': count_reasons(cases),
    }


def visualize(sample: Sample, answer: Answer) -> Dict[str, np.ndarray]:
    vis = sample.captcha.images['full'].copy()
    cv2.putText(vis, f'est={answer.values["theta"]:.2f} gt={sample.gt["theta"]:.2f}',
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return {'vis': vis}


register(TypeSpec(
    name='rotation',
    description='旋转：估计内盘相对正确位置的偏转角',
    generate=generate,
    solver=RotationSolver(),
    metric=metric,
    aggregate=aggregate,
    visualize=visualize,
))
