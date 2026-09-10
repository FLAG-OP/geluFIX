# geluFIX — FlagGems gelu(tanh) 家族 CANN pow 类型 KeyError 修复

修复 [FlagGems issue #5867](https://github.com/flagos-ai/FlagGems/issues/5867)：在昇腾 NPU 上
开启 flag_gems 后，`torch.nn.functional.gelu(x, approximate="tanh")` 编译期直接崩溃：

```
triton.compiler.errors.CompilationError
KeyError: (triton.language.float32, triton.language.int32)
```

一句话版本：**CANN 的 libdevice `pow` 查表只认 float/float 组合，而 kernel 里
`pow(x, 2)` 的字面量 `2` 被 Triton 定型为 int32——查不到就 KeyError。把指数写成
`2.0`，问题消失。gelu 前向/反向/原地版 + geglu + gelu_and_mul 共 8 处同病，一起修。**

```
修复前: gelu(tanh) 全 dtype 编译期 KeyError (CANN 8.5 / 9.0 均复现)
修复后: fp32/fp16/bf16 前向+反向与 CPU 原生 allclose, max|diff| ≤ 7.6e-6
```

## 问题是怎么回事

issue 的最小复现：

```python
import torch, torch_npu, flag_gems
x = torch.rand(5).to('npu:0')
y1 = torch.nn.functional.gelu(x, approximate="tanh")   # enable 前: OK
flag_gems.enable()
y2 = torch.nn.functional.gelu(x, approximate="tanh")   # CompilationError!
```

tanh 近似的 gelu 公式里有 `pow(x, 2)`（平方项）。flag_gems 的 `pow` 来自
`tl_extra_shim`，在 Ascend 上解析到 `triton/language/extra/cann/libdevice.py` 的
extern 实现——一个按操作数 dtype 组合查表的分发器：

```python
return core.extern_elementwise("", "", [arg0, arg1], {
    (fp32, fp32): ("__hmf_powf", fp32),
    (fp16, fp16): ("__hmf_powDh", fp16),
    (bf16, bf16): ("__hmf_powDb", bf16),
}, ...)   # ← 只有 float/float 组合, 没有任何 int 指数条目
```

Python 字面量 `2` 在 Triton 语义里定型为 `int32`，于是 `(fp32, int32)` 查表
KeyError，整个 kernel 编译失败。**最小对照实验**（`tests/pow_dtype_probe.py`，
同一 kernel 只换指数写法）坐实因果：

| 写法 | 结果 |
|---|---|
| `pow(x, 2)`（int） | CompilationError |
| `pow(x, 2.0)`（float） | OK，数值 allclose |
| `pow(x, tl.full((), 2.0, fp32))`（显式） | OK，数值 allclose |

为什么 `approximate="none"` 不崩？它走 `erf`，公式里没有 pow——与 issue 现象
（只有 tanh 崩）完全吻合。

## 修复思路

指数从 int 改 float：`pow(x, 2)` → `pow(x, 2.0)`。

- **不改任何数学**：`2` 与 `2.0` 数值相同，只修类型路由——`2.0` 让查表落到
  `(fp32, fp32)` 条目上
- **跨后端安全**：CUDA 等其他后端的 libdevice pow 对 float 指数同样合法
- kernel 结构、公式、精度零改动

受影响文件（grep 全仓库 `pow(.*, 整数字面量)` 摸底 + 逐个实测）：

| 文件 | int 指数处数 | 症状 |
|---|---|---|
| `ops/gelu.py`（issue 本体） | 4 | gelu tanh 前向 fp32/fp16/bf16、backward、gelu_ 全崩 |
| `fused/geglu.py` | 3 | geglu 崩 |
| `fused/gelu_and_mul.py` | 1 | fused gelu_and_mul(tanh) 崩 |
| `ops/weightnorm.py` + `fused/weight_norm.py`（`pow(,3)`） | 4 | 同模式，**未修**（超出本 issue，见边界） |

## 怎么用

前置：昇腾环境（本修复在 Ascend 910 + CANN 8.5 / torch 2.10 / torch_npu 2.10
+ flag_gems 5.3.5 验证；issue 报告 CANN 9.0 同病）。

**部署**（三个文件）：

```bash
for f in ops/gelu.py; do
  cp /path/to/site-packages/flag_gems/$f /path/to/backup/
done
cp src/gelu.py        /path/to/site-packages/flag_gems/ops/gelu.py
cp src/geglu.py       /path/to/site-packages/flag_gems/fused/geglu.py
cp src/gelu_and_mul.py /path/to/site-packages/flag_gems/fused/gelu_and_mul.py
find /path/to/site-packages/flag_gems -name __pycache__ -exec rm -rf {} +
```

或者 `git apply gelu.patch`（基于 v5.3.5，含全部三个文件与署名标记）。

**验证**（零依赖，直接 python 跑）：

```bash
ASCEND_LAUNCH_BLOCKING=1 python tests/repro_5867.py        # issue 原复现，全 OK
ASCEND_LAUNCH_BLOCKING=1 python tests/gelu_red.py          # 7 项: TOTAL: 7, failed: 0
ASCEND_LAUNCH_BLOCKING=1 python tests/fused_red.py        # 3 项: TOTAL: 3, failed: 0
ASCEND_LAUNCH_BLOCKING=1 python tests/test_upstream_layer.py  # 上游层: 10 项 (含 weightnorm)
ASCEND_LAUNCH_BLOCKING=1 python tests/int_pow_standalone.py   # int-pow kernel 单独跑 (不 enable)
ASCEND_LAUNCH_BLOCKING=1 python tests/accuracy.py         # 精度矩阵全 allclose
ASCEND_LAUNCH_BLOCKING=1 python tests/pow_dtype_probe.py  # 根因三对照
```

**仓库回归**（注意 `--ref cpu`，见下）：

```bash
cd /root/FlagGems
ASCEND_LAUNCH_BLOCKING=1 python -m pytest tests/test_gelu.py \
    tests/test_geglu.py tests/test_gelu_and_mul.py --ref cpu -q
# 144 passed, 0 failed
```

## 精度说明（重要：pytest 失败 ≠ 算子不准）

不传 `--ref cpu` 时 `tests/test_gelu.py` 有 39 个 **none 路径**失败——这不是
本修复的回归，也不是算子精度差，是**参考值被污染**：

- pytest 默认 `TO_CPU=False`，参考值在 **NPU 上算 fp64 gelu**
- 910 没有原生 fp64 AI core，NPU fp64 是软件模拟/降精度实现，参考值自身带误差
- gems kernel 是 fp32 计算，与失真的 fp64 参考比对，fp16 用例的 atol=1e-4 挡不住

证据（`tests/tolerance_probe2.py` + `tests/ulp_fix2.py`）：

| 参考值来源 | 结果 |
|---|---|
| **CPU fp64**（真金标准） | **100% 通过**（FlagGems 容差 / PyTorch 默认容差 / 1-ULP 判定全过） |
| NPU fp64（pytest 默认） | 39 用例失败（修复前原版同样失败，git stash 对照过） |

实际精度（vs CPU fp64）：fp32 max diff 9.5e-7（≈1 ULP）、fp16 全部 ≤1 ULP、
bf16 在 dtype 分辨率内。tanh（本修复）路径任何容差下全过。ULP 归一化分析：
fp16 差异点 100% ≤1 ULP（纯舍入边界翻转），fp32 绝对差全在 1e-7 量级。

**结论**：跑 gelu 相关测试请带 `--ref cpu`。建议上游把 fp64 参考固定到 CPU
（影响所有用 `to_reference(upcast=True)` 且涉及超越函数的测试）。

## 目录结构

```
├── README.md              # 本文
├── gelu.patch             # 三文件最小 diff（含署名），git apply 用
├── libdevice_pow_promotion.patch  # 上游层修复: cann libdevice pow int 指数提升(泛化方案, 见报告§8)
├── src/
│   ├── gelu.py(.orig)     # ops/gelu.py 修复版 + v5.3.5 原版
│   ├── geglu.py(.orig)    # fused/geglu.py 同上
│   └── gelu_and_mul.py(.orig)
├── tests/
│   ├── repro_5867.py      # issue 复现 + 修复验收
│   ├── gelu_red.py        # TDD 红转绿: gelu 家族 7 项
│   ├── fused_red.py       # TDD 红转绿: geglu/gelu_and_mul 3 项
│   ├── test_upstream_layer.py   # 上游层泛用性 10 项 (Part A 子进程 + Part B 叠加态)
│   ├── int_pow_standalone.py    # int-pow kernel 独立脚本 (不 enable, 规避 §8.6 现象)
│   ├── accuracy.py        # 精度矩阵 (3 dtype × 2 模式 + backward)
│   ├── pow_dtype_probe.py # 根因三对照 (int/float/显式 fp32 指数)
│   ├── ulp_fix2.py        # ULP 归一化精度分析
│   ├── tolerance_probe2.py# 容差公式三套判定 (CPU fp64 参考)
│   └── test_gelu_repo.py  # 仓库版 pytest (需放回仓库 tests/ 跑)
├── tools/
│   └── apply_upstream_patch.py  # 上游补丁幂等安装/卸载/状态 (apply|revert|status)
└── docs/
    └── GELU_5867_REPORT.md # 完整报告: 根因、影响面、精度与容差分析、上游层验证(§8)
```

## 如果你想深究 bug 在哪一行

| 原版位置（src/*.orig） | 函数 | 与本次修复的关系 |
|---|---|---|
| gelu.py L43 | `gelu_tanh` 的 `pow(x_fp32, 2)` | **肇事行**（issue 栈指向） |
| gelu.py L54 | `gelu_backward_none` 的 `pow(scale1*x, 2)` | 同病 |
| gelu.py L67/69 | `gelu_backward_tanh` 的两处 `pow(, 2)` | 同病 |
| geglu.py L66/122/126/128 | geglu 前向/反向 | 同病（主动排查） |
| gelu_and_mul.py L74 | `gelu_tanh_and_mul_kernel` | 同病（主动排查） |
| gelu.py L35 | `gelu_none` 的 erf 路径 | 未动（无 pow，不受影响） |

真正的错误源头在 `triton/language/extra/cann/libdevice.py` 的 `pow` 表（上游
triton-ascend 仓库），但 flag_gems 侧传 float 指数是更小、跨后端安全的修法。

## 两种修复层级（可独立使用，也可叠加）

**gelu.patch（flag_gems 层，默认推荐）**：8 处 `pow(x, 2)` → `pow(x, 2.0)`。
最小、跨后端安全，但只修已知调用点。

**libdevice_pow_promotion.patch（triton-ascend 层，泛化方案）**：直接给
`triton/language/extra/cann/libdevice.py` 的 `pow` 加 int 指数自动提升
（constexpr[int]/裸 int → float）。一处修复让**所有** `pow(x, N)` 写法免疫
——包括 weightnorm 的 `pow(norm, 3)` 等 4 处我们没在 gelu.patch 里修的同模式
（原版 flag_gems + 仅打此补丁，全部实测通过，见报告 §8.3）。代价：改动在
triton 环境的 site-packages，换环境要重打；上游合入后此补丁可废弃。

```bash
# 上游层补丁用法 (或用等价的幂等工具)
python tools/apply_upstream_patch.py apply    # 自动探测 triton 路径, 备份 .orig
python tools/apply_upstream_patch.py status   # 查看状态
python tools/apply_upstream_patch.py revert   # 恢复原版
# 或手动:
cd /path/to/triton && patch -p1 < libdevice_pow_promotion.patch
```

## 已知边界（不装完美）

- **weightnorm 的 `pow(norm, 3)` 在 gelu.patch 里未修**（4 处同模式）：gelu.patch
  层面超出本 issue 范围；但 **libdevice_pow_promotion.patch 已覆盖它**
  （原版 weightnorm + 上游补丁实测通过）——要泛化保护就用上游层补丁
- **CANN 9.0 未实测**：issue 报障环境是 CANN 9.0，本机 8.5 复现并修复；
  KeyError 机制（查表）与 CANN 版本无关，判断同修，但无 9.0 实机
- **NPU fp64 参考值问题只诊断未修**：属于测试基础设施问题，影响面超出 gelu
- backward 精度只测了 tanh 路径（none 路径 backward 依赖 erf 同参考值问题）

## 验证矩阵汇总

| 验证项 | 结果 |
|---|---|
| issue #5867 原复现（修复前） | tanh 全 dtype CompilationError（KeyError (fp32,int32)） |
| 根因三对照（int/float/显式指数） | int 崩、float OK 且数值正确——因果坐实 |
| TDD 红→绿（gelu 家族 7 项 + fused 3 项） | 红 5+2 FAIL → **绿 10/10** |
| 精度矩阵（3 dtype × none/tanh + backward） | 与 CPU 原生 allclose 全过，max diff ≤7.6e-6 |
| 仓库 pytest（--ref cpu） | **144 passed / 0 failed** |
| 仓库 pytest（默认 NPU fp64 参考） | 39 none 失败（存量，修复前后一致，已排除回归） |
| ULP 分析 | fp16 差异 100% ≤1 ULP；fp32 绝对差 ~1e-7 |
