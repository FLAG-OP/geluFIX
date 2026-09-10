"""ULP 归一化修正: 输出量级~0.5, ULP 应按输出口径算; 且排除 ref==0 附近的除零伪影。"""
import torch
import torch_npu
import flag_gems

flag_gems.enable()

def ulp_of(dtype_bits, v):
    e = torch.floor(torch.log2(v.abs().clamp_min(1e-8)))
    return torch.pow(2.0, e - dtype_bits)

for dt, bits, label in [(torch.float16, 10, "fp16"), (torch.float32, 23, "fp32")]:
    x = torch.randn(1024, 1024, dtype=torch.float32).to(dt).to("npu:0")
    ref = torch.nn.functional.gelu(x.cpu(), approximate="none")
    got = torch.nn.functional.gelu(x, approximate="none").cpu()
    diff = (ref.float() - got.float()).abs()
    mm = diff > 0
    print(f"[{label}] mismatch {mm.sum()}/{ref.numel()}  max|diff|={diff.max():.3e}")
    if mm.any():
        out_scale = ref[mm].float().abs().clamp_min(2.0 ** -(bits - 1))  # 输出口径
        ulps = diff[mm] / ulp_of(bits, out_scale)
        print(f"   ULP(输出): max={ulps.max():.2f}  <=1: {(ulps<=1).float().mean()*100:.1f}%  <=2: {(ulps<=2).float().mean()*100:.1f}%")
