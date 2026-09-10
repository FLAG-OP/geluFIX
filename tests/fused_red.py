"""gelu_and_mul / geglu 红测试: 同模式 pow(x, 2) int 指数。"""
import sys

import torch
import torch_npu
import flag_gems

flag_gems.enable()
results = []

def check(name, fn):
    try:
        fn()
        results.append((name, True))
        print(f"  [PASS] {name}")
    except Exception as e:
        results.append((name, False))
        print(f"  [FAIL] {name}: {type(e).__name__}: {str(e)[:70]}")

x = torch.randn(4, 8).to("npu:0")
y = torch.randn(4, 8).to("npu:0")

# flag_gems 暴露的 gelu_and_mul (fused)
try:
    from flag_gems.fused.gelu_and_mul import gelu_and_mul
    check("fused gelu_and_mul tanh", lambda: gelu_and_mul(x, y, approximate="tanh"))
    check("fused gelu_and_mul none", lambda: gelu_and_mul(x, y, approximate="none"))
except ImportError as e:
    print("  (skip fused gelu_and_mul import:", e, ")")

# geglu
try:
    from flag_gems.fused.geglu import geglu
    check("fused geglu", lambda: geglu(x, y))
except ImportError as e:
    print("  (skip geglu import:", e, ")")

n_fail = sum(1 for _, c in results if not c)
print(f"\nTOTAL: {len(results)}, failed: {n_fail}")
sys.exit(1 if n_fail else 0)
