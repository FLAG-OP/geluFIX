"""数值精度: gelu 全模式 vs CPU 原生, fp32 用 allclose(1e-6), 半精度放宽。"""
import torch
import torch_npu
import flag_gems

flag_gems.enable()

xs = torch.linspace(-6, 6, 401)
for dt, tol in [(torch.float32, 1e-6), (torch.float16, 2e-3), (torch.bfloat16, 2e-2)]:
    xd = xs.to(dt).to("npu:0")
    for apx in ("none", "tanh"):
        ref = torch.nn.functional.gelu(xd.cpu(), approximate=apx)
        got = torch.nn.functional.gelu(xd, approximate=apx).cpu()
        # gems tanh 内部 fp32 计算; 原生 CPU 也 fp32。对比相同 dtype 语义
        diff = (ref.float() - got.float()).abs().max().item()
        ok = torch.allclose(ref, got, rtol=tol, atol=tol)
        print(f"[{str(dt):16s} {apx:4s}] allclose={ok}  max|diff|={diff:.3e}")

# backward 数值: autograd 对比解析参考 (CPU autograd 为参考)
x = torch.linspace(-4, 4, 101, requires_grad=True)
y = torch.nn.functional.gelu(x, approximate="tanh")
y.sum().backward()
ref_grad = x.grad

xg = torch.linspace(-4, 4, 101).to("npu:0").requires_grad_()
yg = torch.nn.functional.gelu(xg, approximate="tanh")
yg.sum().backward()
got_grad = xg.grad.cpu()
print(f"[backward tanh ] allclose={torch.allclose(ref_grad, got_grad, rtol=1e-5, atol=1e-6)}  "
      f"max|diff|={(ref_grad - got_grad).abs().max().item():.3e}")
