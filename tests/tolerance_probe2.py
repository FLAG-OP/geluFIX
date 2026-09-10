"""容差公式验证 v2: 修 CPU/NPU 混用。"""
import torch
import torch_npu
import flag_gems

flag_gems.enable()

x_npu = torch.randn(1024, 1024, dtype=torch.float32).to(torch.float16).to("npu:0")
got = torch.nn.functional.gelu(x_npu, approximate="none").cpu()
# 参考: fp64 计算后降档 fp16 (模拟 to_reference upcast 流程)
ref64 = torch.nn.functional.gelu(x_npu.cpu().float().to(torch.float64), approximate="none")
ref = ref64.to(torch.float16)

diff = (ref.float() - got.float()).abs()
refabs = ref.float().abs()

fg_pass = (diff <= 1e-4 + 1e-3 * refabs).float().mean()
pt_pass = (diff <= 1e-5 + 1e-3 * refabs).float().mean()
ulp_pass = (diff <= 4.88e-4).float().mean()
print(f"fp16 none, max|diff|={diff.max():.3e}, |ref|中位数={refabs.median():.3f}:")
print(f"  FlagGems 容差(atol=1e-4, rtol=1e-3): 通过 {fg_pass*100:.2f}%")
print(f"  PyTorch 默认(atol=1e-5, rtol=1e-3):   通过 {pt_pass*100:.2f}%")
print(f"  1-ULP 判定(<=4.88e-4):               通过 {ulp_pass*100:.2f}%")
