# -*- coding: utf-8 -*-
"""geo_captcha：几何验证码统一 Benchmark（纯传统图像算法）。

三件套与题型解耦：generate / Solver.solve / metric，详见 core 与 README。
"""
__version__ = '0.1.0'

__all__ = ['Captcha', 'Sample', 'Answer', 'Solver', 'TypeSpec', '__version__']


def __getattr__(name):
    if name in ('Captcha', 'Sample', 'Answer', 'Solver', 'TypeSpec'):
        from . import core
        return getattr(core, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
