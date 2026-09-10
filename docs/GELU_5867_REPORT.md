# FlagGems #5867：gelu(tanh) 家族 CANN pow 类型 KeyError — 分析与修复报告

- **Issue**: [flagos-ai/FlagGems#5867](https://github.com/flagos-ai/FlagGems/issues/5867)
- **代码基线**: FlagGems v5.3.5（issue 报 `5.3.5.post1.dev50+gefa7b6773`，本地
  v5.3.5 的 gelu.py 与 issue 引用行号一致）
- **实验环境**: Ascend 910（8 卡）、CANN 8.5（issue 为 9.0，见 §4 边界）、
  torch 2.10.0+cpu、torch_npu 2.10.0、Python 3.11.15、flag_gems 5.3.5
- **报告日期**: 2026-09-10

---

## 1. 问题

### 1.1 现象

```python
flag_gems.enable()
x = torch.rand(5).to('npu:0')
torch.nn.functional.gelu(x, approximate="tanh")
# → triton.compiler.errors.CompilationError
#   KeyError: (triton.language.float32, triton.language.int32)
```

`approximate="none"` 不受影响（与 issue 一致）。

### 1.2 根因：Triton 类型系统 × CANN libdevice 查表的组合缺陷

```
gelu.py L43: pow(x_fp32, 2)
  → tl_extra_shim.pow
    → (Ascend) triton/language/extra/cann/libdevice.py 的 extern pow
      → 查表 {(fp32,fp32),(fp16,fp16),(bf16,bf16)}   ← 仅 float/float
      → 字面量 2 定型为 int32 → (fp32, int32) 无条目
      → KeyError → CompilationError（编译期，任何 dtype 输入都崩）
```

注：shim 的 `_patch_missing_symbols` 只在符号**缺失**时才打 fallback
（`hasattr(module, "pow")` 为真），符号存在但 dtype 组合残缺时不会介入——
这是该缺陷能漏到运行时的结构性原因。

### 1.3 根因三对照实验（`tests/pow_dtype_probe.py`）

同一最小 kernel，只换指数写法：

| 指数写法 | Triton 定型 | 结果 |
|---|---|---|
| `2` | int32 | **CompilationError**（复现根因） |
| `2.0` | fp32 | OK，与 `x**2` allclose |
| `tl.full((), 2.0, fp32)` | fp32（显式） | OK，allclose |

### 1.4 层次结论

| 层 | 是否有问题 | 依据 |
|---|---|---|
| flag_gems `ops/gelu.py` 等 3 文件的 `pow(, 2)` 写法 | **有（本修复层）** | int 指数触发查表缺口 |
| triton-ascend cann libdevice `pow` 表 | 有（上游根治层） | 表缺 (fp, int) 组合或未做类型提升 |
| flag_gems shim `_patch_missing_symbols` | 有（设计局限） | 只防符号缺失，不防 dtype 组合残缺 |
| CANN / torch_npu | 无 | 编译期 Python 异常，未到硬件 |

### 1.5 影响面（grep `pow(.*, 整数字面量)` + 逐个实测）

| 文件 | 处数 | 实测（修复前） |
|---|---|---|
| `ops/gelu.py` | 4 | tanh 前向 fp32/fp16/bf16 崩；backward 崩；gelu_ 崩 |
| `fused/geglu.py` | 3 | geglu 崩 |
| `fused/gelu_and_mul.py` | 1 | fused gelu_and_mul(tanh) 崩 |
| `ops/weightnorm.py` / `fused/weight_norm.py` | 4（`pow(,3)`） | 未测（路径可达性未确认） |

---

## 2. 修复

### 2.1 改动

三个文件共 8 处：`pow(x, 2)` → `pow(x, 2.0)`（weightnorm 的 `pow(,3)` 未动，
见边界）。每处带 `wt-2026-09-10-fix (#5867)` 标记 + `wt <wangt635@ustc.edu.cn>` 署名。

数值语义零变化（2 与 2.0 相等），只修类型路由。跨后端安全：float 指数在
CUDA/其他后端 libdevice 同样合法。

### 2.2 为什么不在上游层修

triton-ascend 的 cann libdevice `pow` 表补 `(fp32, int32)` 条目或做隐式提升
是根治，但属另一仓库；且 int 指数语义有歧义（`pow(int, int)` 是整数幂？），
上游怎么修需要讨论。flag_gems 侧传 float 指数是最小、无歧义、跨后端的修法，
两层不冲突。

---

## 3. 实验结果（全部真实 NPU 执行）

| # | 验证项 | 方式 | 结果 |
|---|---|---|---|
| 1 | 基线复现 | `repro_5867.py` | tanh 全 dtype CompilationError（KeyError (fp32,int32)），none OK——与 issue 一致 |
| 2 | 根因三对照 | `pow_dtype_probe.py` | int 崩 / float OK / 显式 OK——因果坐实 |
| 3 | TDD 红（gelu 家族） | `gelu_red.py`（修复前） | 5 FAIL（tanh×3 dtype + backward + gelu_） |
| 4 | TDD 绿（gelu 家族） | 同上（修复后） | **7/7** |
| 5 | TDD 红绿（fused） | `fused_red.py` | 红 2 FAIL → 绿 **3/3** |
| 6 | 精度矩阵 | `accuracy.py` | 3 dtype × none/tanh + backward，与 CPU 原生 allclose，max diff ≤7.6e-6 |
| 7 | 仓库 pytest（--ref cpu） | test_gelu/geglu/gelu_and_mul | **144 passed / 0 failed / 15 skipped** |
| 8 | 仓库 pytest（默认参考） | 同上不带 flag | 39 none 失败——存量（见 §5） |
| 9 | 回归排除 | git stash 原版重跑 #8 同失败 | mismatch 比例 7.2% vs 7.2%，非本修复引入 |

---

## 4. 诚实自查：模糊 / 近似 / 跳过 / 暂缓清单

### ⚠️ 已声明的近似

1. **CANN 9.0 未实测**：issue 环境 9.0，本机 8.5。KeyError 是 Python 查表
   机制，与 CANN 版本无关，判断同修；无 9.0 实机验证。
2. **weightnorm `pow(,3)` 未修未测**：同模式写法，但其调用链是否实际到达
   extern pow 未验证（weightnorm 的 shim 解析路径需单独确认），且超出本
   issue 报障范围。列为上游建议。

### 🚫 有意跳过

3. **triton-ascend 上游根治**：见 §2.2，分层决策。
4. **none 路径 backward 精度**：依赖 erf + NPU fp64 参考，受 §5 问题干扰，
   未单独构造 CPU 参考验证（tanh backward 已验证）。

### ❌ 过程中修正过的错误（记录以防复现误导）

5. fused 红测试第一轮"修完仍红"——因为测试 import 的是 site-packages 副本
   而改动在 repo 源码树。同步两处后绿。教训：flag_gems 双副本环境（repo +
   site-packages）每次改动必须同步验证。
6. ULP 分析第一版用输入量级归一化，得出"30 万 ULP"的荒谬值；改为按输出口径
   归一化并剔除近零参考点后，fp16 差异 100% ≤1 ULP。第一版数字作废。
7. 容差探针第一版把 CPU 参考和 NPU 结果在同一表达式混合，触发
   "Pointer argument cannot be accessed from Triton"——修复后数据才有效。

---

## 5. 附带发现：NPU fp64 参考值污染 pytest（存量，与算子无关）

**现象**：默认参考（NPU fp64）下 `test_gelu.py` 39 个 none 用例失败；
`--ref cpu`（CPU fp64）下 **108/108 全过**。

**机制**：`to_reference(upcast=True)` 只转 dtype 不转设备（conftest 默认
`TO_CPU=False`）→ 参考值在 NPU 上算 fp64 gelu → 910 无原生 fp64 AI core，
软件模拟的 fp64 参考自身带误差（erf 类超越函数最明显）→ fp16 用例
atol=1e-4 挡不住参考抖动 → 假失败。

**量化**（`tolerance_probe2.py`，CPU fp64 参考下三套判定）：
FlagGems 容差通过率 100%、PyTorch 默认容差 100%、1-ULP 判定 100%。

**影响面**：所有 `to_reference(upcast=True)` 且参考含超越函数的测试在
Ascend 上都可能假失败。建议上游：fp64 参考固定在 CPU 执行，或对 NPU
fp64 软件模拟加告警。

**本报告的精度结论全部以 CPU fp64 为参考**（`accuracy.py` / `ulp_fix2.py`）。

---

## 6. 复现 / 验证命令速查

```bash
cd /root/geluFIX
ASCEND_LAUNCH_BLOCKING=1 python tests/repro_5867.py
ASCEND_LAUNCH_BLOCKING=1 python tests/gelu_red.py       # 7/7
ASCEND_LAUNCH_BLOCKING=1 python tests/fused_red.py      # 3/3
ASCEND_LAUNCH_BLOCKING=1 python tests/accuracy.py
ASCEND_LAUNCH_BLOCKING=1 python tests/pow_dtype_probe.py

cd /root/FlagGems
ASCEND_LAUNCH_BLOCKING=1 python -m pytest tests/test_gelu.py \
  tests/test_geglu.py tests/test_gelu_and_mul.py --ref cpu -q   # 144 passed
```

## 7. 建议后续

1. diff（gelu.patch）整理为 PR 提交上游，引用 issue #5867。
2. 上游顺带修 weightnorm 的 `pow(,3)`（同模式，4 处）。
3. triton-ascend 侧考虑 cann libdevice pow 的 int 指数隐式提升或更完整的
   dtype 表 + 错误信息（报出支持的组合，而不是裸 KeyError）。
4. 测试基础设施：fp64 参考固定 CPU（§5），避免 Ascend 全量测试假失败。
