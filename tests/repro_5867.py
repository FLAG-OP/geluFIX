"""issue #5867 复现: gelu approximate='tanh' 在 enable 后。"""
import torch
import torch_npu
import flag_gems

x = torch.rand(5).to("npu:0")
y1 = torch.nn.functional.gelu(x, approximate="tanh")
print("native gelu tanh:", y1.tolist())

flag_gems.enable()
try:
    y2 = torch.nn.functional.gelu(x, approximate="tanh")
    print("gems gelu tanh:  ", y2.tolist())
    print("allclose:", torch.allclose(y1, y2))
except Exception as e:
    print(f"GEMS FAILED: {type(e).__name__}: {str(e)[:200]}")

# 对照: approximate='none'
try:
    y3 = torch.nn.functional.gelu(x, approximate="none")
    print("gems gelu none:   OK")
except Exception as e:
    print(f"GELU NONE FAILED: {type(e).__name__}: {str(e)[:150]}")

# 对照: fp16/bf16
for dt in (torch.float16, torch.bfloat16):
    xd = torch.rand(5).to("npu:0").to(dt)
    try:
        yd = torch.nn.functional.gelu(xd, approximate="tanh")
        print(f"gems gelu tanh {dt}: OK")
    except Exception as e:
        print(f"gems gelu tanh {dt}: FAILED {type(e).__name__}: {str(e)[:100]}")
