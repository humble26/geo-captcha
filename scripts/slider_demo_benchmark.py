# -*- coding: utf-8 -*-
"""基准测试：准确性 A/B 与吞吐量对比。

1) 准确性：同一批样本上，以页面真值 (gapX, gapY) 为基准，对比
   - 旧管线：find_gap(refine=False, ensemble=False)，单方法过门槛、整数像素
   - 新管线：find_gap()（默认），集成决策 + 形态学清理 + 亚像素
2) 吞吐量：分别以 --workers 1 与 --workers N 跑 solver，对比墙钟时间。

用法:
  python benchmark.py                                # 全部：200 样本 + 2×80 实例
  python benchmark.py --samples 100 --trials 60 --workers 6
  python benchmark.py --accuracy-only / --throughput-only
"""
import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent
SOLVER = ROOT / 'solver.py'

from slider_captcha.detector import find_gap          # noqa: E402
from slider_captcha.page_driver import GRAB_JS, decode_image  # noqa: E402
from slider_captcha.runner import demo_url            # noqa: E402


def bench_accuracy(n):
    """抓 n 个样本（含真值），离线对比新旧定位管线的 x 误差。"""
    from playwright.sync_api import sync_playwright

    legacy, modern = [], []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(demo_url())
        pg.wait_for_function('window.captchaSeq >= 1')
        for i in range(n):
            seq = pg.evaluate('window.captchaSeq')
            d = pg.evaluate(GRAB_JS, ['#bgCanvas', '#pieceCanvas'])
            gx, _gy = pg.evaluate('[gapX, gapY]')     # 真值仅用于评测，求解器不读它
            bg = decode_image(d['bg'])
            piece = decode_image(d['piece'])
            old = find_gap(bg, piece, refine=False, ensemble=False)
            new = find_gap(bg, piece)
            legacy.append(old.x - gx)
            modern.append(new.x - gx)
            pg.evaluate('newChallenge()')
            pg.wait_for_function(f'window.captchaSeq > {seq}')
        b.close()
    _report_errors('旧管线(单方法+整数像素)', legacy)
    _report_errors('新管线(集成决策+亚像素)  ', modern)
    return legacy, modern


def _report_errors(name, errs):
    ab = [abs(e) for e in errs]
    exact = sum(e < 0.5 for e in ab)
    within1 = sum(e <= 1.0 for e in ab)
    within_tol = sum(e <= 4.0 for e in ab)
    print(f'{name}  平均|误差|={statistics.mean(ab):.3f}px  最大={max(ab):.2f}px  '
          f'误差<0.5px: {exact}/{len(ab)}  ≤1px: {within1}/{len(ab)}  '
          f'≤4px(判定容差): {within_tol}/{len(ab)}')


def bench_throughput(trials, workers):
    """子进程跑 solver，对比串行与并发的墙钟时间。"""
    rows = []
    for w in (1, workers):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(SOLVER), '--trials', str(trials), '--workers', str(w)],
            capture_output=True, text=True, encoding='utf-8')
        dt = time.perf_counter() - t0
        m = re.search(r'成功率: (\d+)/(\d+)', proc.stdout)
        ok, total = (int(m.group(1)), int(m.group(2))) if m else (0, trials)
        rows.append((w, dt, ok, total))
    base = rows[0][1]
    for w, dt, ok, total in rows:
        print(f'workers={w}: 墙钟 {dt:.1f}s  {ok}/{total} 成功  '
              f'{trials / dt * 60:.0f} 例/分' + (f'  (加速 {base / dt:.1f}x)' if w != 1 else '  (基线)'))
    return rows


def main():
    ap = argparse.ArgumentParser(description='滑块求解器基准测试')
    ap.add_argument('--samples', type=int, default=200, help='准确性对比样本数')
    ap.add_argument('--trials', type=int, default=80, help='吞吐对比的每轮实例数')
    ap.add_argument('--workers', type=int, default=6, help='并发档位')
    ap.add_argument('--accuracy-only', action='store_true')
    ap.add_argument('--throughput-only', action='store_true')
    a = ap.parse_args()

    if not a.throughput_only:
        print(f'== 准确性 A/B（{a.samples} 个样本，以页面真值为基准） ==')
        bench_accuracy(a.samples)
        print()
    if not a.accuracy_only:
        print(f'== 吞吐量（每档 {a.trials} 实例） ==')
        bench_throughput(a.trials, a.workers)


if __name__ == '__main__':
    main()
