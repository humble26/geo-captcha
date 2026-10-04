# -*- coding: utf-8 -*-
"""题型：点选——按语序点击指定文字，按顺序输出各目标中心坐标。

生成器-求解器契约（合成数据的公平性约定，README 同步说明）：
  - 目标字与干扰字共用一个色相族 h0±14°，背景色相避开 h0±40°——
    求解器允许利用 meta 中的 hue_family 做色相分割（真实点选验证码也靠颜色/形状先验）；
  - 字体路径在 meta 中给出（合成数据字体已知；真实样本需另配 OCR 方案）；
  - 提示语（"按顺序点击 …"）放在 meta['prompt']，画布左上角同步渲染仅供人工查看。
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .. import synth
from ..core import (Answer, Captcha, Sample, Solver, TypeSpec, count_reasons,
                    dist2d, error_stats, register)

W, H = 320, 180
TOL_PX = 12.0
CHAR_POOL = list('山川木火土水日月天云风花雪松石泉海波林森岩峰岛港城')


@dataclass
class Difficulty:
    k: int = 4                  # 目标字数
    distractors: int = 1        # 干扰字数（同色族、非目标）
    rot_max: float = 18.0       # 字符旋转角上限（度）
    size_range: Tuple[int, int] = (30, 42)
    noise: float = 0.4


PRESETS = {
    'easy': Difficulty(k=3, distractors=0, rot_max=0.0, noise=0.0),
    'normal': Difficulty(),
    'hard': Difficulty(k=5, distractors=3, rot_max=28.0, size_range=(26, 46),
                       noise=0.8),
}


def _place_boxes(k_total, sizes, rng) -> List[Tuple[int, int]]:
    """拒绝采样放置互不重叠的字符框，返回中心点。"""
    boxes: List[Tuple[int, int]] = []
    for s in sizes:
        for _ in range(400):
            cx = int(rng.integers(s // 2 + 8, W - s // 2 - 8))
            cy = int(rng.integers(s // 2 + 8, H - s // 2 - 8))
            if all(abs(cx - bx) > (s + bs) / 2 + 6 or abs(cy - by) > (s + bs) / 2 + 6
                   for bx, by, bs in boxes):
                boxes.append((cx, cy, s))
                break
        else:
            boxes.append((int(rng.integers(20, W - 20)), int(rng.integers(20, H - 20)), s))
    return [(b[0], b[1]) for b in boxes]


def generate(seed: int, difficulty: Difficulty = None) -> Sample:
    d = difficulty or Difficulty()
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    font = synth.find_font()
    if font is None:
        raise RuntimeError('未找到可用的中文字体（C:\\Windows\\Fonts），点选题无法生成')

    hue0 = float(rng.integers(0, 360))
    img = synth.texture_background(W, H, rng, richness=0.9 * (1 + d.noise),
                                   avoid_hue=hue0, hue_margin=40, noise=d.noise)

    chars = prng.sample(CHAR_POOL, d.k + d.distractors)
    prompt, distractor_chars = chars[:d.k], chars[d.k:]

    sizes = [int(rng.integers(*d.size_range)) for _ in range(d.k + d.distractors)]
    centers = _place_boxes(len(sizes), sizes, rng)
    order = list(range(len(sizes)))
    prng.shuffle(order)                      # 屏上位置与语序解耦
    gt_centers: List[Optional[List[float]]] = [None] * d.k

    for idx in order:
        is_target = idx < d.k
        ch = (prompt + distractor_chars)[idx]
        s = sizes[idx]
        color = synth.hsv_bgr((hue0 + rng.uniform(-14, 14)) % 360,
                              int(rng.integers(150, 220)),
                              int(rng.integers(60, 110)))
        angle = float(rng.uniform(-d.rot_max, d.rot_max))
        sprite = synth.render_text_sprite(ch, font, s, color, angle)
        cx, cy = centers[idx]
        synth.paste_sprite(img, sprite, cx, cy)
        if is_target:
            # GT 取可见墨迹中心（字体上下不对称会使贴图中心偏离墨迹中心，
            # 以墨迹为准才能与求解器从图像测得的中心对齐）
            alpha = sprite[:, :, 3]
            ys, xs = np.where(alpha > 10)
            ink_cx = cx - sprite.shape[1] // 2 + (xs.min() + xs.max() + 1) / 2
            ink_cy = cy - sprite.shape[0] // 2 + (ys.min() + ys.max() + 1) / 2
            gt_centers[idx] = [float(ink_cx), float(ink_cy)]

    # 左上角提示语（人工查看用；求解器读 meta['prompt']）
    tip = '按顺序点击: ' + ' '.join(prompt)
    tip_sprite = synth.render_text_sprite(tip, font, 15, (120, 120, 120))
    synth.paste_sprite(img, tip_sprite, 2 + tip_sprite.shape[1] // 2,
                       2 + tip_sprite.shape[0] // 2)

    captcha = Captcha(type='click', images={'full': img},
                      meta={'prompt': prompt, 'font': font,
                            'hue_family': hue0, 'rot_max': d.rot_max,
                            'size_range': list(d.size_range),
                            'difficulty': asdict(d)},
                      seed=seed)
    gt = {'prompt': prompt, 'centers': gt_centers}
    return Sample(captcha=captcha, gt=gt)


class ClickSolver(Solver):
    """色相分割 -> 膨胀后连通域（防多笔画断字拆散）-> 字形模板匹配（尺度/旋转
    归一，先 128x128 归一再轻膨胀 3x3x1 保持笔画细节）-> 全局最优唯一指派。

    消融记录（150 官方种子）：匹配期重膨胀 5x5x2 会抹掉相似字（岩/峰/石）的
    区分性笔画，改为 3x3x1 + 128px 后全对 132→135、平均偏差 3.5→1.9px；
    贪心换全局最优指派 +1；分象限余弦、倒角距离均无增益。剩余失败为二值
    字形匹配的固有混淆平台（见 README 失败模式）。
    """

    SEG_DILATE_K = np.ones((5, 5), np.uint8)      # 连通域用：合并断笔
    SEG_DILATE_IT = 2
    MATCH_DILATE_K = np.ones((3, 3), np.uint8)    # 匹配用：尺寸归一后轻膨胀
    MATCH_DILATE_IT = 1
    RES = 128
    N_ANGLES = 13
    THR = 0.45

    def solve(self, captcha: Captcha) -> Answer:
        meta = captcha.meta
        img = captcha.images['full']
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h0 = float(meta['hue_family']) / 2.0          # cv2 色相单位
        dh = 20.0 / 2.0
        hch = (hsv[:, :, 0].astype(int) - int(round(h0))) % 180
        hdist = np.minimum(hch, 180 - hch)
        mask = ((hdist <= dh) & (hsv[:, :, 1] >= 60) &
                (hsv[:, :, 2] >= 50) & (hsv[:, :, 2] <= 235)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # 在膨胀掩膜上取连通域：把"川"等多笔画断开的字并成一个组件
        big = cv2.dilate(mask, self.SEG_DILATE_K, iterations=self.SEG_DILATE_IT)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(big, 8)
        smin, smax = meta['size_range']
        blobs = []
        for i in range(1, n):
            x, y, bw, bh, _area = stats[i]
            if bw < smin * 0.5 or bh < smin * 0.5 or bw > smax * 2.2 or bh > smax * 2.2:
                continue
            comp = ((labels[y:y + bh, x:x + bw] == i) & (mask[y:y + bh, x:x + bw] > 0))
            blobs.append({'cx': float(x + bw / 2), 'cy': float(y + bh / 2),
                          'patch': comp})

        refs = {}
        for ch in meta['prompt']:
            refs[ch] = self._ref_masks_cached(ch, meta['font'],
                                              float(meta.get('rot_max', 0.0)))

        K, B = len(meta['prompt']), len(blobs)
        # 补丁预处理（缩放+膨胀+向量化）每 blob 只做一次，供 K 个提示字复用
        crops = [self._prep_patch(b['patch']) for b in blobs]
        scores = np.zeros((K, B))
        for i, ch in enumerate(meta['prompt']):
            ref_mat, ref_norms = refs[ch]                     # (A, D), (A,)
            for j, cvec in enumerate(crops):
                if cvec is None:
                    continue
                per_angle = (ref_mat @ cvec[0]) / (ref_norms * cvec[1])
                scores[i, j] = float(per_angle.max())

        # 全局最优唯一指派：K≤6 且 B≤10 时枚举全部单射（P(B,K) ≤ 15120）
        assign = [None] * K
        if 0 < B and K <= B and B <= 10 and K <= 6:
            best_perm, best_sum = None, -1.0
            for perm in itertools.permutations(range(B), K):
                tot = sum(scores[i, perm[i]] for i in range(K))
                if tot > best_sum:
                    best_sum, best_perm = tot, perm
            for i, j in enumerate(best_perm):
                if scores[i, j] >= self.THR:
                    assign[i] = j

        centers, missing = [], []
        for i, ch in enumerate(meta['prompt']):
            if assign[i] is not None:
                b = blobs[assign[i]]
                centers.append([b['cx'], b['cy']])
            else:
                centers.append(None)
                missing.append(ch)
        return Answer(values={'centers': centers, 'chars': list(meta['prompt'])},
                      debug={'n_blobs': len(blobs), 'missing': missing,
                             'scores': scores.tolist()})

    _REF_CACHE: Dict[tuple, tuple] = {}

    @classmethod
    def _ref_masks_cached(cls, ch: str, font: str, rot_max: float):
        key = (ch, font, round(rot_max, 2))
        if key not in cls._REF_CACHE:
            cls._REF_CACHE[key] = cls._ref_masks(ch, font, rot_max)
        return cls._REF_CACHE[key]

    @classmethod
    def _prep_patch(cls, patch: np.ndarray):
        """拼块补丁 -> (归一化前向量, L2 范数)；空补丁返回 None。"""
        ys, xs = np.where(patch)
        if len(xs) == 0:
            return None
        crop = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
        crop = cv2.resize(crop, (cls.RES, cls.RES), interpolation=cv2.INTER_AREA)
        crop = cv2.dilate(crop, cls.MATCH_DILATE_K, iterations=cls.MATCH_DILATE_IT)
        vec = crop.astype(np.float32).ravel()
        return vec, float(np.linalg.norm(vec))

    @classmethod
    def _ref_masks(cls, ch: str, font: str, rot_max: float):
        """提示字参考掩码 -> (角度堆叠矩阵 (A, D), 各行 L2 范数)。多旋转角，
        先归一化到 RES x RES 再做与补丁同强度的轻膨胀（膨胀放在尺寸归一之后，
        保证相对笔画粗细一致）。"""
        angles = np.linspace(-rot_max, rot_max, cls.N_ANGLES) if rot_max > 0 else [0.0]
        masks = []
        for ang in angles:
            sprite = synth.render_text_sprite(ch, font, 48, (255, 255, 255), float(ang))
            m = (sprite[:, :, 3] > 128).astype(np.uint8) * 255
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                continue
            m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            m = cv2.resize(m, (cls.RES, cls.RES), interpolation=cv2.INTER_AREA)
            masks.append(cv2.dilate(m, cls.MATCH_DILATE_K, iterations=cls.MATCH_DILATE_IT))
        if not masks:
            zero = np.zeros((1, cls.RES * cls.RES), np.float32)
            return zero, np.ones(1, np.float32)
        mat = np.stack([m.astype(np.float32).ravel() for m in masks])
        return mat, np.linalg.norm(mat, axis=1)


def metric(answer: Answer, gt: Dict) -> Dict:
    centers = answer.values.get('centers') or []
    gt_centers = gt['centers']
    if len(centers) != len(gt_centers) or any(c is None for c in centers):
        return {'pass': False, 'order_exact': False, 'dev_mean': None,
                'fail_reason': '漏识', 'n_targets': len(gt_centers)}
    devs = [dist2d(c, g) for c, g in zip(centers, gt_centers)]
    dev_mean = float(np.mean(devs))
    exact = all(dv <= TOL_PX for dv in devs)
    return {'pass': exact, 'order_exact': exact, 'dev_mean': dev_mean,
            'dev_max': float(max(devs)),
            'fail_reason': None if exact else '点错位置',
            'n_targets': len(gt_centers)}


def aggregate(cases: List[Dict]) -> Dict:
    exact = [c for c in cases if c['order_exact']]
    devs = [c['dev_mean'] for c in cases if c.get('dev_mean') is not None]
    s = error_stats(devs)
    return {
        'n': len(cases),
        'order_exact_rate': len(exact) / len(cases),
        'dev_mean_of_mean': s['mean'], 'dev_median': s['median'],
        'dev_p95': s['p95'], 'dev_max': s['max'],
        'tol_px': TOL_PX,
        'fail_reasons': count_reasons(cases),
    }


def visualize(sample: Sample, answer: Answer) -> Dict[str, np.ndarray]:
    vis = sample.captcha.images['full'].copy()
    for c in answer.values.get('centers') or []:
        if c is not None:
            cv2.circle(vis, (int(c[0]), int(c[1])), 6, (0, 0, 255), 2)
    for c in sample.gt['centers']:
        cv2.drawMarker(vis, (int(c[0]), int(c[1])), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
    return {'vis': vis}


register(TypeSpec(
    name='click',
    description='点选：按语序输出各目标字中心坐标',
    generate=generate,
    solver=ClickSolver(),
    metric=metric,
    aggregate=aggregate,
    visualize=visualize,
))
