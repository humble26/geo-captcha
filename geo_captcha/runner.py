# -*- coding: utf-8 -*-
"""评测 runner：与题型解耦的批量评测、结果落盘与 markdown 报告。"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .core import TypeSpec, all_specs, derive_seed, jsonable, load_types
from .synth import ascii_hist

MAX_FAIL_IMAGES = 10


def _difficulty_for(spec: TypeSpec, preset: Optional[str]):
    if not preset or preset == 'default':
        return None
    mod = __import__(f'geo_captcha.types.{spec.name}', fromlist=['PRESETS'])
    presets = getattr(mod, 'PRESETS', {})
    if preset not in presets:
        raise KeyError(f'{spec.name} 无难度档 {preset}（可选: {sorted(presets)}）')
    return presets[preset]


def _difficulty_json(difficulty) -> Optional[dict]:
    return asdict(difficulty) if is_dataclass(difficulty) else None


def run_type(spec: TypeSpec, n: int, master_seed: int,
             difficulty=None, out_dir: Optional[Path] = None) -> Dict:
    """评测单个题型 n 例，返回 {'name','summary','cases'}。"""
    cases: List[Dict] = []
    fail_saved = 0
    for i in range(n):
        seed = derive_seed(master_seed, spec.name, i)
        sample = spec.generate(seed, difficulty)
        t0 = time.perf_counter()
        answer = spec.solver.solve(sample.captcha)
        elapsed = (time.perf_counter() - t0) * 1000.0
        m = spec.metric(answer, sample.gt)
        rec = {
            'index': i, 'seed': seed, 'type': spec.name,
            'difficulty': _difficulty_json(difficulty),
            'metrics': jsonable(m), 'answer': jsonable(answer.values),
            'gt': jsonable(sample.gt), 'time_ms': round(elapsed, 2),
        }
        cases.append(rec)

        if not m.get('pass') and out_dir is not None and fail_saved < MAX_FAIL_IMAGES \
                and spec.visualize is not None:
            fail_dir = out_dir / 'failures' / spec.name
            fail_dir.mkdir(parents=True, exist_ok=True)
            for name, vis in spec.visualize(sample, answer).items():
                cv2.imwrite(str(fail_dir / f'case{i:04d}-seed{seed}-{name}.png'), vis)
            fail_saved += 1

    return {'name': spec.name, 'summary': spec.aggregate([c['metrics'] for c in cases]),
            'cases': cases}


def _run_type_worker(payload):
    """进程池 worker：按题型名重建规格并评测（参数均可 pickle）。"""
    name, n, master_seed, preset, out_dir = payload
    print(f'[{name}] 评测 {n} 例 …', flush=True)
    load_types()
    spec = all_specs()[name]
    if not spec.available:
        return {'name': name, 'summary': {
            'unavailable': True, 'reason': spec.unavailable_reason}, 'cases': []}
    difficulty = _difficulty_for(spec, preset)
    return run_type(spec, n, master_seed, difficulty,
                    Path(out_dir) if out_dir else None)


def run_all(type_names: Optional[List[str]], n: int, master_seed: int,
            preset: Optional[str] = None, out_dir: Optional[str] = None,
            jobs: int = 1) -> Dict:
    """评测多个题型；type_names 为空时跑全部可用题型。

    jobs > 1 时按题型开进程池并行（题型间零依赖，种子逐例派生，
    并行不影响结果）。
    """
    load_types()
    specs = all_specs()
    names = type_names or [k for k, v in specs.items() if v.available]
    unknown = [t for t in names if t not in specs]
    if unknown:
        raise KeyError(f'未知题型: {unknown}（可用: {sorted(specs)}）')

    out = Path(out_dir) if out_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    payloads = [(name, n, master_seed, preset, str(out) if out else None)
                for name in names]
    if jobs > 1 and len(payloads) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(jobs, len(payloads))) as ex:
            blocks = list(ex.map(_run_type_worker, payloads))
    else:
        blocks = [_run_type_worker(p) for p in payloads]
    return {'master_seed': master_seed, 'n': n, 'preset': preset,
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'env': _env_info(), 'blocks': blocks}


def _env_info() -> Dict:
    import sys
    info = {'python': sys.version.split()[0]}
    try:
        import cv2
        info['opencv'] = cv2.__version__
    except Exception:                          # noqa: BLE001
        pass
    try:
        import PIL
        info['pillow'] = PIL.__version__
    except Exception:                          # noqa: BLE001
        pass
    try:
        import slider_captcha
        info['slider_captcha'] = slider_captcha.__version__
    except Exception:                          # noqa: BLE001
        info['slider_captcha'] = None
    return info


def write_outputs(result: Dict, out_dir: str) -> Path:
    """落盘 results.json + report.md，返回报告路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=1),
        encoding='utf-8')
    report = render_markdown(result)
    (out / 'report.md').write_text(report, encoding='utf-8')
    return out / 'report.md'


def render_markdown(result: Dict) -> str:
    lines: List[str] = []
    env = result['env']
    lines.append('# 几何验证码 Benchmark 报告')
    lines.append('')
    lines.append(f'- 生成时间：{result["generated_at"]}｜主种子：{result["master_seed"]}'
                 f'｜每题型例数：{result["n"]}｜难度档：{result["preset"] or "default"}')
    lines.append(f'- 环境：Python {env.get("python")}｜OpenCV {env.get("opencv")}'
                 f'｜Pillow {env.get("pillow")}｜slider_captcha {env.get("slider_captcha")}')
    lines.append('')

    for block in result['blocks']:
        name = block['name']
        summary = block['summary']
        lines.append(f'## {name}')
        lines.append('')
        if summary.get('unavailable'):
            lines.append(f'> 题型不可用：{summary.get("reason")}')
            lines.append('')
            continue
        lines.append('| 指标 | 数值 |')
        lines.append('|---|---|')
        for k, v in summary.items():
            if k == 'fail_reasons':
                continue
            if isinstance(v, float):
                lines.append(f'| {k} | {v:.4f} |')
            elif isinstance(v, dict):
                lines.append(f'| {k} | {json.dumps(v, ensure_ascii=False)} |')
            else:
                lines.append(f'| {k} | {v} |')
        lines.append('')

        cases = block['cases']
        key = _primary_error_key(name)
        if key:
            errs = [c['metrics'][key] for c in cases if c['metrics'].get(key) is not None]
            label = '误差分布（px）' if key.endswith('px') else \
                '误差分布（deg）' if key.endswith('deg') else '指标分布'
            lines.append(f'**{label}**')
            lines.append('```')
            lines.extend(ascii_hist(errs, *(_hist_range(name))))
            lines.append('```')
            lines.append('')

        fails = [c for c in cases if not c['metrics'].get('pass')]
        if fails:
            lines.append(f'**失败案例（{len(fails)} 例）**')
            lines.append('')
            lines.append('| 例号 | seed | 归因 | 关键数值 | 耗时ms |')
            lines.append('|---|---|---|---|---|')
            for c in fails[:20]:
                m = c['metrics']
                kv = {k: round(v, 3) for k, v in m.items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool)
                      and k != 'pass'}
                lines.append(f"| {c['index']} | {c['seed']} | {m.get('fail_reason')} "
                             f"| {kv} | {c['time_ms']} |")
            if len(fails) > 20:
                lines.append(f'| …（其余 {len(fails) - 20} 例见 results.json）| | | | |')
            lines.append('')
        lines.append('---')
        lines.append('')
    return '\n'.join(lines)


def _primary_error_key(type_name: str) -> Optional[str]:
    return {'slider': 'err_px', 'click': 'dev_mean', 'rotation': 'err_deg'}.get(type_name)


def _hist_range(type_name: str):
    return {'slider': (0.0, 2.0), 'click': (0.0, 30.0), 'rotation': (0.0, 15.0)}.get(
        type_name, (0.0, 1.0))
