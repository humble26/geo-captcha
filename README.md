# 几何验证码工具集

**一个仓库、两个顶层包、单向依赖**：

- [`slider_captcha/`](slider_captcha/) —— **求解库**：OpenCV 缺口定位 + 拟人轨迹 +
  可插拔页面驱动，可被宿主程序（如 Token 领取助手）直接嵌入；
- [`geo_captcha/`](geo_captcha/) —— **统一评测框架**：滑块/点选/旋转/空间推理四题型
  三件套解耦评测，合成数据逐例种子可复现，传统图像算法求解。

> ⚠️ 仅供学习研究，以及对你拥有或已获授权测试的系统使用。请遵守目标站点的服务条款与相关法律法规。

![demo](docs/demo.png)

## 正式成绩单（geo_captcha，4×200 例，seed 20261003）

`python -m geo_captcha run --types slider,click,rotation,spatial --n 200 --seed 20261003`

| 题型 | n | 主指标 | 主要失败模式 |
|---|---|---|---|
| slider | 200 | **通过率 100%（200/200），定位误差均值 0.285px**（验收线 ≤0.5px） | 无 |
| click | 200 | 语序全对率 **91.0%**（通过例偏差中位 0.43px） | 漏识 11（低置信主动放弃）、点错位置 7 |
| rotation | 200 | ±3° 通过率 **98.0%**，MAE 1.14° | 歧义图案 4（margin 可事前识别） |
| spatial | 200 | 完全还原率 **97.0%** | 全局错乱 6（低纹理场景） |

滑块题型复用 `slider_captcha.find_gap`，在统一 harness 下复现并超越原基线
（原独立 demo：100%、0.456px）。逐例数据与失败案例图在 `runs/`（已 gitignore，
一条命令可再生）。

## 模块化架构

```
geo_captcha（评测框架）
  generate(seed, difficulty) ─► Sample(captcha, gt)
  Solver.solve(captcha) ─► Answer
  metric(answer, gt) ─► 逐例指标 ─► aggregate ─► report.md + results.json
        │
        └── types/slider.py ──调用──► slider_captcha.find_gap（唯一的跨包依赖，惰性导入）
slider_captcha（求解库）
  detector（缺口定位） ── trajectory（拟人轨迹） ── page_driver（可插拔页面驱动）
        └── runner（自带浏览器跑批） ── doctor（环境自检）
```

- 依赖方向单一：评测框架 → 求解库，求解库不知道评测框架的存在；
- `slider_captcha` 只依赖 numpy+opencv（cv2 惰性导入），playwright 仅浏览器
  入口需要——宿主缺哪层只影响哪层，`import slider_captcha` 任何环境不炸；
- 新增题型 = 实现 generate / Solver / metric 三件 + 一行注册，runner 零改动。

## 目录结构

```
15-几何验证码/
├── slider_captcha/          求解库
│   ├── detector.py          缺口定位（可独立 CLI：python -m slider_captcha.detector）
│   ├── trajectory.py        拟人轨迹 + 动作编排（零第三方依赖）
│   ├── page_driver.py       Async/Sync Playwright 适配器 + 共享求解流程
│   ├── runner.py            自带浏览器批量跑批 CLI
│   ├── doctor.py            环境自检
│   └── demo/index.html      本地演示验证码（随包分发）
├── geo_captcha/             评测框架
│   ├── core.py              Captcha/Sample/Answer/TypeSpec/注册表/种子派生
│   ├── synth.py             合成渲染公共层
│   ├── runner.py            批量评测 + results.json + report.md + 失败案例落盘
│   ├── __main__.py          CLI（python -m geo_captcha）
│   └── types/               slider / click / rotation / spatial 四题型
├── scripts/                 slider_demo_benchmark.py（demo 页 A/B 与吞吐基准）
├── examples/sync_solve.py   同步 API 嵌入示例
├── tests/                   29 个单测（含数值等价性回归防护）
├── docs/                    demo 截图、《集成任务提示词》
├── solver.py / gap_detector.py  兼容 CLI 壳（= python -m slider_captcha.runner / .detector）
├── pyproject.toml           单一发行版 geo-captcha，同时提供两个包
└── requirements.txt
```

## 快速开始

```bash
pip install -e .                       # 单仓库一次装齐 slider_captcha + geo_captcha
pip install -e ".[browser]"            # 浏览器求解可选装
python -m playwright install chromium  # 首次需下载 Chromium 内核

python -m slider_captcha.runner --trials 200 --workers 6   # 滑块求解端到端
python -m geo_captcha run --types slider,click,rotation,spatial --n 200 \
    --seed 20261003 --jobs 4                               # 统一评测（约 7.5s）
python -m slider_captcha.doctor                            # 环境自检
python -m unittest discover -s tests                       # 单测
```

兼容入口壳保留：`python solver.py`（= slider_captcha.runner）、
`python gap_detector.py bg.png piece.png --debug`（= slider_captcha.detector）。

## slider_captcha 求解库

解滑块验证码只需回答两个问题：**拖多远**、**怎么拖**。

1. **拖多远 —— 缺口定位**（`detector.py`）
   - 主力 `dark_shape`：背景转"暗度图"（暗于 Otsu 阈值越多值越大，连续灰度），
     与拼图块 alpha 形状做相关匹配，在平滑响应峰上做抛物线亚像素插值。
     注意：亚像素插值只在连续响应上有效，二值掩码上的插值实测有害（误差 +40%）；
   - 互证候选：二值 Otsu、自适应阈值、浅色描边轮廓、边缘模板、掩码 SQDIFF；
   - 集成决策：过门槛候选按"3px 内互证"聚类，簇分 = 最高分 + 0.05×(簇员数-1)，
     胜出簇内取优先级最高成员（不做均值融合——描边方法的系统偏差会拖累均值）；
   - 拿不到拼图块图片时退化为 `edge_only`（准确率有限）。

2. **怎么拖 —— 拟人轨迹**（`trajectory.py`）
   smoothstep 加减速曲线，带随机过冲、回调、抖动与偶尔停顿；
   动作序列与传输层解耦，同步/异步驱动共用同一份编排。

3. **执行拖拽**（`page_driver.py`）
   画布像素与页面像素按"显示宽度/图像宽度"自动换算（高分屏无需改动）；
   批量运行 asyncio worker 池并发、事件驱动换题、失败偏差 EMA 校准。

嵌入宿主（同步 API 示例，完整见 `examples/sync_solve.py`）：

```python
from slider_captcha import solve_on_page_sync, Selectors

result = solve_on_page_sync(page)          # page 为宿主已登录流程中的页面
if result.ok:
    ...                                    # result.gap.x / result.distance
```

适配真实站点：换 `Selectors`（bg/piece/slider 选择器 + solved/error/ready JS 探针），
不改库代码。**非浏览器宿主**（桌面窗口截图 + SendInput 拖拽）：
`find_gap` 接受任意 ndarray；实现驱动接口的 8 个小方法即可复用全流程。

## geo_captcha 评测框架

### 题型契约与求解器要点

- **slider**：生成器与浏览器 demo 同构（压暗缺口 + 浅色描边 + 明亮拼图块）；
  描边宽 1px、alpha 标定 0.45（消融：无描边偏置 +0.84px、硬描边 -0.56px）；
- **click**：目标字与干扰字同色相族、背景避开该族；GT 取**可见墨迹中心**；
  求解器 HSV 分割 → 膨胀连通域（"川"类断笔字合并）→ 参考字与补丁先归一化
  128x128 再轻膨胀 3x3x1 → 余弦多角度匹配 → 全局最优唯一指派；
- **rotation**：内盘 = 原场景旋转 theta；求解器极坐标展开后**在接缝两侧各偏移
  2..8px**（径向间隔 4px）采样，循环移位 SSD 用 FFT 一次算齐 + 抛物线细化。
  实测教训：采样带径向间隔 12px 时跨环内容相关性弱（MAE 11°），收到 4px 后 0.11°；
- **spatial**：3x3 拼图；边界代价 = 接缝加权 SSD + **双向**线性预测项（Pomeranz 式），
  按拼块自身边界内部变差归一化（MGC 简化版）——归一化是决定性的一步
  （不归一 30 例 12 对，归一后 30/30；对称 sqrt(var_a·var_b) 实测有害）；
  9! = 362880 排列全枚举向量化求全局最优（~35ms/例）。

### 当前失败模式（诚实呈现）

- **click 漏识 11 + 点错 7**：相似字形（岩/峰/石）二值字形匹配的固有混淆平台；
  改进方向：灰度/软边缘特征、完整 MGC、提示语 OCR；
- **rotation 歧义图案 4**：低纹理/近对称场景的更深层代价谷（margin 事前可识别）；
- **spatial 全局错乱 6**：低 richness 场景归一化代价仍不足；改进方向：完整 MGC。

### 运行成本（4×200 例）

| 阶段 | 墙钟 | 手段 |
|---|---|---|
| 优化前 | 18.1s | — |
| 算法优化后 | 14.5s | click 补丁预处理每 blob 一次 + 矩阵化余弦；rotation FFT 循环互相关；spatial 全对代价广播化 + float32 |
| `--jobs 4` 后 | **7.5s** | 按题型进程池并行（结果与串行逐字段一致） |

### 被证伪的方向（防走回头路）

旋转题按外带梯度加权 / 角度方差归一（1.7%~8.3%）、远离接缝的多带联合；
拼图对称归一化 sqrt(var_a·var_b)（55→32）、多条带（无增益）；
点选分象限余弦、倒角距离（无增益）。

### 真实样本入口

`core.load_real_samples(dir)` 读取 `dir/<题型>/*.json`
（`{"images": {...}, "meta": {...}, "gt": {...}}`，图像相对路径），
便于以后混入实拍样本 + JSON 人工标注。

## 环境验证

- Python 3.10.11 与 3.14.6 双解释器验证（29 个单测 + 全部 CLI）；
- 两解释器均为 OpenCV 5.0 / Pillow 12；
- 集成计划：见 `docs/集成任务提示词.md`（供接入 Token 领取助手）。

## 许可证

[MIT](LICENSE)
