"""Part A 独立脚本: 干净进程 (不 enable) 的 int 指数 pow kernel 直接验证。

被 test_upstream_layer.py 以子进程方式调用; 也可单独运行:
  ASCEND_LAUNCH_BLOCKING=1 python tests/int_pow_standalone.py
"""
import torch
import torch_npu
import triton
import triton.language as tl
from triton.language.extra import cann


@triton.jit
def _k(x_ptr, o_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    y = cann.libdevice.pow(x, 2)
    tl.store(o_ptr + offs, y)


def main():
    x = torch.rand(8, device="npu:0")
    o = torch.empty_like(x)
    _k[(1,)](x, o, N=8)
    torch.npu.synchronize()
    assert torch.allclose(o, x * x), f"数值错误: {o} vs {x*x}"
    print("PART_A_OK int-pow kernel (no enable): values correct")


if __name__ == "__main__":
    main()
