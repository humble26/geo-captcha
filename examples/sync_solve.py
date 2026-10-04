# -*- coding: utf-8 -*-
"""同步 API 嵌入示例：在宿主已有的 sync Playwright 页面上解滑块。

这是把求解器集成进外部程序的标准姿势之一（同步版）；
异步版用 solve_on_page(await page)，两者参数一致。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 未 pip install 时引导包路径

from playwright.sync_api import sync_playwright

from slider_captcha import solve_on_page_sync
from slider_captcha.runner import demo_url


def main():
    with sync_playwright() as p:
        # 实际集成时：这里通常是宿主已经打开并登录中的页面，而不是新启动的浏览器
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(demo_url())
        page.wait_for_function('window.captchaSeq >= 1')

        for i in range(3):
            r = solve_on_page_sync(page)
            print(f'第 {i + 1} 次: {"成功" if r.ok else "失败"} '
                  f'gap.x={r.gap.x:.1f} ({r.gap.method}, {r.gap.score:.2f})')
            page.evaluate('newChallenge()')
            page.wait_for_timeout(300)

        browser.close()


if __name__ == '__main__':
    main()
