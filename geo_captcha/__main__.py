# -*- coding: utf-8 -*-
"""python -m geo_captcha：统一评测 CLI。

用法：
  python -m geo_captcha list
  python -m geo_captcha run --types slider,click,rotation,spatial --n 200 \
      --seed 20261003 --out runs/main --difficulty normal
"""
from __future__ import annotations

import argparse
import sys


def _reconfigure():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def main():
    _reconfigure()
    ap = argparse.ArgumentParser(prog='geo_captcha',
                                 description='几何验证码统一 Benchmark')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('list', help='列出注册的题型与可用性')

    run = sub.add_parser('run', help='批量评测')
    run.add_argument('--types', default=None,
                     help='逗号分隔题型名，缺省为全部可用题型')
    run.add_argument('--n', type=int, default=100, help='每题型例数')
    run.add_argument('--seed', type=int, default=20261003, help='主种子')
    run.add_argument('--difficulty', default=None,
                     help='难度档 easy/normal/hard，缺省为各题型默认难度')
    run.add_argument('--out', default='runs/latest', help='输出目录')
    run.add_argument('--jobs', type=int, default=1,
                     help='按题型并行的进程数（题型间独立，不影响结果）')
    run.add_argument('--quiet', action='store_true', help='不打印逐题型进度')

    args = ap.parse_args()

    from .core import all_specs, load_types
    from .runner import run_all, write_outputs

    if args.cmd == 'list':
        load_types()
        for name, spec in all_specs().items():
            status = '可用' if spec.available else f'不可用（{spec.unavailable_reason}）'
            print(f'{name:<10} {status}  {spec.description}')
        return

    types = [t.strip() for t in args.types.split(',') if t.strip()] \
        if args.types else None
    result = run_all(types, args.n, args.seed, args.difficulty,
                     out_dir=None if args.quiet else args.out, jobs=args.jobs)
    report_path = write_outputs(result, args.out)
    print()
    _print_console_summary(result)
    print(f'\n逐例 JSON: {args.out}/results.json')
    print(f'报告: {report_path}')


def _print_console_summary(result):
    for block in result['blocks']:
        name, s = block['name'], block['summary']
        if s.get('unavailable'):
            print(f'{name:<10} 不可用：{s["reason"]}')
            continue
        keys = []
        for k in ('pass_rate', 'order_exact_rate', 'exact_rate'):
            if k in s:
                keys.append(f'{k}={s[k]:.1%}')
        for k in ('err_mean', 'mae', 'dev_mean_of_mean', 'tile_acc_mean'):
            if k in s and s[k] == s[k]:
                keys.append(f'{k}={s[k]:.3f}')
        print(f'{name:<10} n={s["n"]}  ' + '  '.join(keys))


if __name__ == '__main__':
    main()
