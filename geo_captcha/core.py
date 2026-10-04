# -*- coding: utf-8 -*-
"""统一评测框架核心。

三件套约定（与题型一一对应，评测 runner 与题型完全解耦）：
  - generate(seed, difficulty) -> Sample       合成生成器（同步产出 ground truth 与难度参数）
  - Solver.solve(captcha) -> Answer            求解器（只见图像与 meta，不见 gt）
  - metric(answer, gt) -> dict                 逐例指标（含 pass / fail_reason 归因）

新增题型 = 实现这三件 + register(TypeSpec)，runner/CLI 零改动。
"""
from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------- 数据结构

@dataclass
class Captcha:
    """验证码实例：图像 + 元信息（不含 ground truth）。"""

    type: str
    images: Dict[str, np.ndarray]      # 命名图像，BGR 或 BGRA
    meta: Dict[str, Any]               # 布局/字体/难度等求解器可见信息
    seed: int                          # 本例种子（生成器据此可精确复现）


@dataclass
class Sample:
    """生成器产出：验证码 + 真值（真值只进评测侧，不进求解器）。"""

    captcha: Captcha
    gt: Dict[str, Any]


@dataclass
class Answer:
    """求解器输出：结构化答案 + 调试信息（debug 供失败归因用）。"""

    values: Dict[str, Any]
    debug: Dict[str, Any] = field(default_factory=dict)


class Solver:
    """求解器协议：实现 solve(captcha) -> Answer 即可。"""

    def solve(self, captcha: Captcha) -> Answer:  # pragma: no cover - 协议定义
        raise NotImplementedError


@dataclass
class TypeSpec:
    """题型规格：把三件套注册进框架所需的一切。"""

    name: str
    description: str
    generate: Callable[..., Sample]                 # (seed, difficulty=None) -> Sample
    solver: Any                                     # Solver 协议实现
    metric: Callable[[Answer, Dict], Dict]          # (answer, gt) -> 逐例指标
    aggregate: Callable[[List[Dict]], Dict]         # 逐例指标列表 -> 题型汇总
    visualize: Optional[Callable] = None            # (sample, answer) -> {名称: BGR 图}，失败案例落盘用
    available: bool = True
    unavailable_reason: str = ''


# ---------------------------------------------------------------- 注册表

_REGISTRY: Dict[str, TypeSpec] = {}


def register(spec: TypeSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f'题型重复注册: {spec.name}')
    _REGISTRY[spec.name] = spec


def get_spec(name: str) -> TypeSpec:
    if name not in _REGISTRY:
        raise KeyError(f'未知题型: {name}（可用: {sorted(_REGISTRY)}）')
    return _REGISTRY[name]


def all_specs() -> Dict[str, TypeSpec]:
    return dict(_REGISTRY)


def load_types() -> None:
    """导入全部题型模块，触发注册（缺依赖的题型会以 available=False 注册）。"""
    from .types import slider, click, rotation, spatial  # noqa: F401


# ---------------------------------------------------------------- 种子与复现

def derive_seed(master: int, type_name: str, index: int) -> int:
    """主种子 -> 逐例种子：跨运行/跨解释器稳定（禁用随机化的 hash()）。"""
    payload = f'{master}:{type_name}:{index}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2 ** 31)


# ---------------------------------------------------------------- 指标工具

def wrap180(deg: float) -> float:
    """角度差归一到 (-180, 180]。"""
    return (float(deg) + 180.0) % 360.0 - 180.0


def dist2d(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def error_stats(errors: List[float]) -> Dict[str, float]:
    """误差分布摘要（输入为空时返回 NaN 值字段）。"""
    if not errors:
        return {'mean': float('nan'), 'median': float('nan'),
                'p95': float('nan'), 'max': float('nan')}
    arr = np.sort(np.asarray(errors, dtype=float))
    q95 = arr[min(len(arr) - 1, int(math.ceil(0.95 * len(arr))) - 1)]
    return {'mean': float(arr.mean()), 'median': float(np.median(arr)),
            'p95': float(q95), 'max': float(arr[-1])}


def count_reasons(cases: List[Dict]) -> Dict[str, int]:
    """统计失败归因分布（含 None -> 通过）。"""
    out: Dict[str, int] = {}
    for c in cases:
        key = c.get('fail_reason') or 'pass'
        out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------- 真实样本入口

def load_real_samples(root: str) -> List[Sample]:
    """加载真实样本（供以后混入实拍数据）。

    目录约定：root/<题型>/<名称>.json，内容：
      {"images": {"bg": "bg.png", ...}, "meta": {...}, "gt": {...}, "seed": 123}
    图像路径相对该 json 文件。seed 缺省时由文件名派生。
    """
    samples: List[Sample] = []
    for json_path in sorted(Path(root).rglob('*.json')):
        data = json.loads(json_path.read_text(encoding='utf-8'))
        type_name = json_path.parent.name
        images = {k: _imread(json_path.parent / v) for k, v in data['images'].items()}
        # 缺省 seed 由文件名的稳定哈希派生（str.__hash__ 跨进程随机化，不可用）
        stable_index = zlib.crc32(json_path.stem.encode('utf-8'))
        seed = data.get('seed')
        seed = int(seed) if seed is not None else derive_seed(0, type_name, stable_index)
        captcha = Captcha(type=type_name, images=images,
                          meta=data.get('meta', {}), seed=seed)
        samples.append(Sample(captcha=captcha, gt=data['gt']))
    return samples


def _imread(path: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'真实样本图像读取失败: {path}')
    return img


# ---------------------------------------------------------------- 序列化

def jsonable(obj: Any) -> Any:
    """numpy/tuple 等 -> 纯 Python 结构，供 JSON 落盘。"""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj
