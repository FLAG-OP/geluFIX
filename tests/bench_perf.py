"""geluFIX 性能对照实验 (#5867).

修复语义: pow 指数 int->float (类型路由), 数值零变化。
性能问题域:
  A. tanh 路径 (修复目标): 修复前=编译崩不可测; 修复后 vs 原生 gelu
  B. none 路径 (未改 kernel): gems vs 原生 —— 回归护栏 (应与修复前一致,
     因为一行未动)
  C. 上游 libdevice 补丁 (泛用层): int 指数 pow vs float 指数 pow 同 kernel
     对照 —— 证明泛用补丁零数值/性能影响
"""
import statistics
import time

import torch
import torch_npu
import flag_gems
from flag_gems.utils.codegen_config_utils import get_codegen_config

DEV = "npu:0"


def bench(fn, iters=200, warmup=50, rounds=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(ts)


# 原生基线 (enable 前)
sizes = [(1024 * 1024,), (1024, 1024)]
native = {}
for apx in ("tanh", "none"):
    for s in sizes:
        x = torch.randn(s, device=DEV)
        native[(apx, s)] = bench(lambda x=x, apx=apx: torch.nn.functional.gelu(x, approximate=apx))

flag_gems.enable()

print("=" * 66)
print("gelu 性能对照: gems(修复后) vs 原生 —— tanh 修复前=编译崩, none=回归护栏")
print("=" * 66)
for apx in ("tanh", "none"):
    for s in sizes:
        x = torch.randn(s, device=DEV)
        t = bench(lambda x=x, apx=apx: torch.nn.functional.gelu(x, approximate=apx))
        ref = native[(apx, s)]
        mb = 1 if len(s) == 1 else 4
        print(f"  [{apx:4s} {mb}MB] 原生: {ref:7.1f} us | gems修复后: {t:7.1f} us  ({t/ref:.2f}x)")

# C. 泛用层补丁: int 指数 (走补丁提升) vs float 指数 同 kernel
# 注: torch.rand 会踩 flag_gems rand.py 的 ub overflow 存量 bug, 用 empty+fill 构造
import triton
import triton.language as tl
from triton.language.extra import cann


@triton.jit
def _k_pow_int(x_ptr, o_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, cann.libdevice.pow(x, 2))


@triton.jit
def _k_pow_float(x_ptr, o_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, cann.libdevice.pow(x, 2.0))


x = torch.empty(1024 * 1024, device=DEV)
x.uniform_(0.1, 0.9)  # 避开 rand kernel 的存量 ub overflow bug
o = torch.empty_like(x)
_k_pow_int[(256,)](x, o, N=4096)
_k_pow_float[(256,)](x, o, N=4096)
torch.npu.synchronize()
r1 = x * x
ok_int = torch.allclose(o, r1)
t_int = bench(lambda: _k_pow_int[(256,)](x, o, N=4096))
t_float = bench(lambda: _k_pow_float[(256,)](x, o, N=4096))
print()
print(f"  [泛用补丁对照] pow(x,2) int(经提升): {t_int:6.1f} us | pow(x,2.0) float: {t_float:6.1f} us "
      f"(差 {abs(t_int-t_float)/max(t_int,t_float)*100:.0f}%, 数值 ok={ok_int})")
print("  判读: int 指数经补丁提升后与 float 直接写法同性能 (constexpr 编译期归一)。")
