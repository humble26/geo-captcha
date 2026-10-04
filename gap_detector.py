# -*- coding: utf-8 -*-
"""兼容入口：缺口定位已迁入 slider_captcha.detector。

    python gap_detector.py bg.png piece.png --debug
等价于
    python -m slider_captcha.detector bg.png piece.png --debug
"""
from slider_captcha.detector import main

if __name__ == '__main__':
    main()
