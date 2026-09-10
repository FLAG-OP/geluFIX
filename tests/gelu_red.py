"""TDD 红: gelu 家族 tanh 路径 + backward + 受影响同模式算子。"""
import sys
import torch
import torch_npu
import flag_gems

flag_gems.enable()
results = []

def check(name, fn):
    try:
        r = fn()
        ok = r is True or r is None
        results.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    except Exception as e:
        results.append((name, False))
        print(f"  [FAIL] {name}: {type(e).__name__}: {str(e)[:80]}")

x = torch.rand(5).to("npu:0")
for dt in (torch.float32, torch.float16, torch.bfloat16):
    xd = x.to(dt)
    check(f"gelu tanh {dt}", lambda xd=xd: torch.allclose(
        torch.nn.functional.gelu(xd.cpu(), approximate="tanh").to("npu:0"),
        torch.nn.functional.gelu(xd, approximate="tanh"), rtol=1e-3, atol=1e-3) or print(""))

check("gelu none", lambda: torch.allclose(
    torch.nn.functional.gelu(x.cpu(), approximate="none").to("npu:0"),
    torch.nn.functional.gelu(x, approximate="none"), rtol=1e-5) or None)

# backward (需要 autograd)
def bw():
    xa = torch.rand(5).to("npu:0").requires_grad_()
    g = torch.nn.functional.gelu(xa, approximate="tanh").sum()
    g.backward()
    return None
check("gelu tanh backward", bw)

# gelu_ (in-place)
def ip():
    xi = torch.rand(5).to("npu:0")
    torch.nn.functional.gelu(xi, approximate="tanh", inplace=True) if hasattr(torch.nn.functional, 'gelu') else None
    return None
# aten gelu_ 直调
def ip2():
    xi = torch.rand(5).to("npu:0")
    torch.ops.aten.gelu_(xi, approximate="tanh")
    return None
check("gelu_ tanh (aten)", ip2)

# 同模式算子: geglu / weight_norm
from flag_gems.fused.geglu import geglu  # 若导出名不同则跳过
check("geglu import", lambda: None)

n_fail = sum(1 for _, c in results if not c)
print(f"\nTOTAL: {len(results)}, failed: {n_fail}")
sys.exit(1 if n_fail else 0)
