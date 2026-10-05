# -*- coding: utf-8 -*-
"""合成渲染公共层：纹理背景、拼图形状掩码、中文/ASCII 文字精灵（PIL）。

所有函数确定性地消费调用方传入的 rng（numpy Generator），
保证同一 seed 下逐像素可复现。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 中文渲染候选字体（按序探测；都没有则退化为 ASCII 池）
# 目录按平台排列：Windows 优先（本机行为不变），Linux 供 CI/移植使用（Noto CJK / 文泉驿正黑）
_FONT_CANDIDATES = ['msyh.ttc', 'msyhbd.ttc', 'simhei.ttf', 'simsun.ttc',
                    'Deng.ttf', 'arial.ttf',
                    'NotoSansCJK-Regular.ttc', 'NotoSansCJKsc-Regular.otf',
                    'wqy-zenhei.ttc']
_FONT_DIRS = [Path('C:/Windows/Fonts'),
              Path('/usr/share/fonts/opentype/noto'),
              Path('/usr/share/fonts/truetype/wqy')]


def find_font() -> Optional[str]:
    for d in _FONT_DIRS:
        for name in _FONT_CANDIDATES:
            p = d / name
            if p.exists():
                return str(p)
    return None


def hsv_bgr(h_deg: float, s: int, v: int) -> Tuple[int, int, int]:
    """角度制 HSV -> BGR 三元组。"""
    h = int(round((h_deg % 360.0) / 2.0))
    bgr = cv2.cvtColor(np.uint8([[[h, int(s), int(v)]]]), cv2.COLOR_HSV2BGR)
    return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])


def hue_dist(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def texture_background(w, h, rng: np.random.Generator, richness=1.0,
                       avoid_hue: Optional[float] = None,
                       hue_margin: float = 40.0, noise=0.0) -> np.ndarray:
    """随机纹理背景：渐变底色 + 圆斑 + 线段 + 高斯噪声。

    avoid_hue 给定时，底色与所有元素色相都避开它 ±hue_margin
    （点选题靠它保证字色可分割）。
    """
    def pick_hue() -> float:
        for _ in range(32):
            hue = float(rng.integers(0, 360))
            if avoid_hue is None or hue_dist(hue, avoid_hue) > hue_margin:
                return hue
        return (float(avoid_hue) + 180.0) % 360.0

    base_hue = pick_hue()
    v0 = int(rng.integers(140, 175))
    v1 = int(rng.integers(175, 205))
    top = np.array(hsv_bgr(base_hue, int(rng.integers(30, 55)), v0), dtype=float)
    bottom = np.array(hsv_bgr((base_hue + rng.integers(-25, 25)) % 360,
                              int(rng.integers(30, 55)), v1), dtype=float)
    img = np.linspace(top, bottom, h, dtype=float)[:, None, :].repeat(w, axis=1)

    n_blobs = int(240 * richness)
    for _ in range(n_blobs):
        hue = pick_hue()
        color = hsv_bgr(hue, int(rng.integers(30, 70)), int(rng.integers(80, 230)))
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(2, 12))
        cv2.circle(img, (cx, cy), r, color, -1, lineType=cv2.LINE_AA)
    for _ in range(int(8 * richness)):
        hue = pick_hue()
        color = hsv_bgr(hue, int(rng.integers(40, 70)), int(rng.integers(70, 160)))
        cv2.line(img,
                 (int(rng.integers(0, w)), int(rng.integers(0, h))),
                 (int(rng.integers(0, w)), int(rng.integers(0, h))),
                 color, int(rng.integers(1, 3)), lineType=cv2.LINE_AA)

    img = np.clip(img, 0, 255).astype(np.uint8)
    if noise > 0:
        g = rng.normal(0, 6.0 * noise, img.shape)
        img = np.clip(img.astype(float) + g, 0, 255).astype(np.uint8)
    return img


def puzzle_mask(w, h, x, y, size=52, bump_r=10) -> np.ndarray:
    """拼图块形状掩码：方形主体 + 顶部凹口 + 右侧凸起（与 slider_captcha demo 一致）。"""
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (x, y), (x + size - 1, y + size - 1), 255, -1, lineType=cv2.LINE_AA)
    notch_r = size * 0.19
    cv2.circle(m, (int(x + size * 0.5), y), int(round(notch_r)), 0, -1, lineType=cv2.LINE_AA)
    cv2.circle(m, (x + size, y + size // 2), bump_r, 255, -1, lineType=cv2.LINE_AA)
    return m


def edge_band(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """掩膜边界带（描边用）。"""
    k = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, k, iterations=width) - cv2.erode(mask, k, iterations=width)


def render_text_sprite(text: str, font_path: str, size: int, color_bgr,
                       angle: float = 0.0) -> np.ndarray:
    """PIL 渲染文字 -> RGBA ndarray（BGR 颜色），可旋转。透明底。"""
    from PIL import Image, ImageDraw, ImageFont
    pad = size // 2 + 2
    img = Image.new('RGBA', (size + pad * 2, size + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size)
    rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text((pad, pad), text, font=font, fill=rgb + (255,))
    if angle:
        img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
    return np.array(img)                     # H x W x 4，通道序 RGBA


def paste_sprite(canvas: np.ndarray, sprite: np.ndarray, cx: int, cy: int) -> None:
    """把 RGBA 精灵按中心点贴到 BGR 画布（alpha 混合，越界裁剪）。"""
    sh, sw = sprite.shape[:2]
    x0, y0 = cx - sw // 2, cy - sh // 2
    x1, y1 = x0 + sw, y0 + sh
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].astype(float)
    region = canvas[cy0:cy1, cx0:cx1].astype(float)
    alpha = sub[:, :, 3:4] / 255.0
    bgr = sub[:, :, 2::-1]                   # RGBA -> BGR
    canvas[cy0:cy1, cx0:cx1] = np.clip(region * (1 - alpha) + bgr * alpha, 0, 255).astype(np.uint8)


def ascii_hist(values: List[float], lo: float, hi: float, bins: int = 12,
               width: int = 36) -> List[str]:
    """误差直方图的 ASCII 行（报告用）。"""
    if not values:
        return ['（无数据）']
    hist, edges = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    peak = max(hist.max(), 1)
    lines = []
    for i, cnt in enumerate(hist):
        bar = '█' * int(round(cnt / peak * width))
        lines.append(f'{edges[i]:>7.2f} ~ {edges[i + 1]:<7.2f} |{bar} {cnt}')
    over = sum(1 for v in values if v > hi)
    if over:
        lines.append(f'  （另有 {over} 例超出 {hi}）')
    return lines
