# -*- coding: utf-8 -*-
"""题型子包：导入即注册全部题型（缺依赖的题型以 available=False 注册）。"""
from __future__ import annotations

from ..core import TypeSpec, register


def _try_register(module_name: str, builder):
    """导入题型模块；其依赖缺失时注册一个 available=False 的占位。"""
    try:
        __import__(f'geo_captcha.types.{module_name}')
    except Exception as e:                     # noqa: BLE001 - 优雅降级
        register(TypeSpec(
            name=module_name,
            description=f'{module_name} 题型（当前环境不可用）',
            generate=None, solver=None, metric=None, aggregate=None,
            available=False,
            unavailable_reason=f'{type(e).__name__}: {e}',
        ))


_try_register('slider', None)
_try_register('click', None)
_try_register('rotation', None)
_try_register('spatial', None)
