"""最小对照: cann libdevice pow 的 dtype 组合行为。
  pow(fp32, 2)      -> 现网 gelu.py 写法 (int 标量)
  pow(fp32, 2.0)    -> float 标量
  pow(fp32, fp32)   -> 全显式
"""
import torch
import torch_npu
import triton
import triton.language as tl
from triton.language.extra import cann as cann_libdevice

pow_fn = cann_libdevice.libdevice.pow


@triton.jit
def k_int_exp(x_ptr, o_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    y = pow_fn(x, 2)
    tl.store(o_ptr + offs, y)


@triton.jit
def k_float_exp(x_ptr, o_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    y = pow_fn(x, 2.0)
    tl.store(o_ptr + offs, y)


@triton.jit
def k_explicit(x_ptr, o_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    y = pow_fn(x, tl.full((), 2.0, tl.float32))
    tl.store(o_ptr + offs, y)


x = torch.rand(8, device="npu:0")
ref = x ** 2

for name, k in [("int 2", k_int_exp), ("float 2.0", k_float_exp), ("explicit fp32", k_explicit)]:
    o = torch.empty_like(x)
    try:
        k[(1,)](x, o, N=8)
        torch.npu.synchronize()
        ok = torch.allclose(o, ref, rtol=1e-5)
        print(f"[{name:14s}] OK  allclose={ok}")
    except Exception as e:
        print(f"[{name:14s}] FAILED: {type(e).__name__}: {str(e)[:90]}")
