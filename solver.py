# -*- coding: utf-8 -*-
"""兼容入口：批量运行器已迁入 slider_captcha.runner。

    python solver.py --trials 200 --workers 6
等价于
    python -m slider_captcha.runner --trials 200 --workers 6
"""
from slider_captcha.runner import main

if __name__ == '__main__':
    main()
