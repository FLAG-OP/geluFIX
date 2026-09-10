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

---

## 8. 追加实验：上游层（triton-ascend libdevice）修复的可行性验证（2026-09-10）

**问题**：能不能不逐个改调用方，直接在 pow 的分发层修，让所有 `pow(x, N)` 写法
天然免疫？——可以，已在本机验证。

### 8.1 调用机制考古（为什么 host 侧 monkey-patch 全部失败）

JIT 编译器（`spec/ascend/compiler/code_generator.py` `call_Function`）对
builtin_namespace 里的 libdevice 函数：先 `_unwrap_if_constexpr(args)` 再
`fn(*args, **kws)` **不注入 `_semantic`**。实测（spy hook）kernel 里写的
`pow(x, 2)` 到达函数体时：`arg1 = constexpr[2]`，`_semantic = None`。

因此 v1–v4 的四轮尝试（host 侧属性替换 / semantic.to_tensor 提升 /
arg1.to() 提升 / 裸 int 判定）全部无效——要么拿不到 semantic，要么 arg1
根本不是裸 int。

### 8.2 有效修复（v5，`libdevice_pow_promotion.patch`）

在 `triton/language/extra/cann/libdevice.py` 的 `pow` 函数体开头加：

```python
if type(arg1) is int:
    arg1 = float(arg1)
elif type(arg1).__name__ == "constexpr":
    v = arg1.value          # constexpr 的 .value 存裸 Python 值
    if type(v) is int:
        arg1 = float(v)
```

`extern_elementwise` 内部的 `to_tensor` 会把 float 定型为 fp32，查表落到
`(fp32, fp32)`。**注意**：fp16 底数 + int 指数场景会提升成 `(fp16, fp32)`
组合——表里同样没有，仍会 KeyError；完整修复应按 arg0.dtype 构造标量。
gelu 家族全部是 fp32 中间量（`x_fp32`），实测不受此限。

### 8.3 泛用性验证（原版 flag_gems + 仅打 libdevice 补丁）

把 flag_gems 三个文件恢复 v5.3.5 原版（int 指数），只留 libdevice 补丁：

| 验证 | 结果 |
|---|---|
| `pow(x, 2)` int 指数最小 kernel | OK，数值 allclose |
| gelu tanh fp32/fp16/bf16（原版 int 指数） | 全 OK，与原生 allclose |
| fused gelu_and_mul / geglu（原版） | OK |
| **weight_norm 的 `pow(norm, 3)`（原版，从未单独修过）** | **OK**——上游层一处修复覆盖了它 |
| gelu_ / backward | OK |
| 仓库 pytest gelu 家族 `--ref cpu` | 144 passed |

**结论**：上游层修复一处，覆盖所有 int 指数调用点（含 weightnorm 这类我们
没在 geluFIX 里修的），泛用性确实更强——用户的直觉正确。

### 8.4 weight_norm pytest 失败的裁定（与补丁无关）

补丁态下 test_weight_norm 偶现 1-2 个 bf16/fp16 用例失败，一度疑似补丁
引入。三轮排查裁定为**存量浮点非确定性**：

1. 同输入 A/B（原版 vs 补丁，同种子）：输出**逐位一致**（三组实验）
2. gems 输出 vs CPU `aten._weight_norm` 同输入：**逐位一致**
   （初版 A/B 的"参考"是笔者手写公式算错——`_weight_norm` 语义是
   `v/||v||*g`，已在报告中修正记录）
3. 原版 libdevice 连跑 pytest 3 次：**同样 1-2 个失败**，且失败用例 ID
   每次不同（dtype0-0 / dtype2--1 / dtype2-0 轮换）——NPU 归约的并行
   顺序非确定，输出在 bf16 1 ULP 边界抖动，atol=1e-4 挡不住

### 8.5 enable() 对裸 libdevice kernel 的编译影响（独立现象，未定论）

泛用性验证过程中发现：同一个最小 int-pow kernel（直接调
`cann.libdevice.pow(x, 2)`），**enable 前编译 OK，`flag_gems.enable()` 之后
编译报 `MLIRCompilationError`（ConvertLinalgIR pass）**。多次复现，稳定。

排查确认：
- 与 int 指数无关——float 指数版在 enable 后同样路径未见测试失败（gelu 全绿）
- 与本补丁无关——原版 libdevice 下同样复现（KeyError 与 MLIR 错误是两个
  不同阶段的失败，前者根本没到 MLIR）
- `pow_dtype_probe.py` 之所以一直 OK：它不 enable，独立进程

推测是 flag_gems 的 libentry/autotune 全局包装改变了裸 Triton kernel 的
编译上下文（pass pipeline 配置），但未深挖——超出本 issue 范围，如实记录。
**实践建议**：验证 libdevice 层行为时用不 enable 的独立进程
（`tests/int_pow_standalone.py` 即为此设计）。

### 8.6 推荐部署

两层修复**可独立也可叠加**（本机当前为叠加态，全绿）：

- **最小改动**（只动 flag_gems）：用 gelu.patch——跨后端安全，但只修 8 处调用点
- **泛化根治**（动 triton 包）：用 libdevice_pow_promotion.patch——覆盖全部
  现有及未来的 int 指数调用（含 weightnorm），但属 triton-ascend 环境的
  site-packages 改动，换环境需重打；上游合入后此补丁可废弃
- 建议上游 PR 走 libdevice 层（泛用），flag_gems 层 8 处 `2.0` 改动作为
  立即可用的热修并行提交
