"""上游层修复验证。

分两部分:
  Part A (子进程, 不 enable): 最小 int 指数 kernel —— libdevice 补丁的直接验证
  Part B (enable 环境): gelu 家族 / weightnorm pow(,3) / fused —— 叠加态功能验证

Part A 必须在未 enable 的干净进程跑: 实测发现 flag_gems.enable() 之后的
Triton 编译上下文 (libentry/autotune 包装) 会让裸 libdevice 调用走不同
pass 路径, 同一 kernel enable 前 OK / enable 后 MLIRCompilationError
(与 int 指数无关的独立现象, 见报告 §8.6)。
"""
import subprocess
import sys

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, True))
        print(f"  [PASS] {name}")
    except Exception as e:
        results.append((name, False))
        print(f"  [FAIL] {name}: {type(e).__name__}: {str(e)[:80]}")


# Part A: 干净进程的 int-pow kernel (@jit 需要真实文件, 用 int_pow_standalone.py)
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def part_a():
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "int_pow_standalone.py")],
                       capture_output=True, text=True, timeout=240)
    assert "PART_A_OK" in r.stdout, f"子进程失败: {r.stderr[-300:]}"


check("pow(x,2) int kernel (干净进程, 上游补丁直接验证)", part_a)

# Part B: enable 环境功能验证 (当前机器为 gelu.patch + 上游补丁叠加态)
import torch  # noqa: E402
import torch_npu  # noqa: E402
import flag_gems  # noqa: E402

flag_gems.enable()

x = torch.rand(5).to("npu:0")
for apx in ("tanh", "none"):
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        def _g(apx=apx, dt=dt):
            xd = x.to(dt)
            ref = torch.nn.functional.gelu(xd.cpu(), approximate=apx)
            got = torch.nn.functional.gelu(xd, approximate=apx).cpu()
            assert torch.allclose(ref, got, rtol=2e-2, atol=2e-2), "数值偏差"
        check(f"gelu {apx} {str(dt).split('.')[-1]}", _g)


def _wn():
    torch.manual_seed(0)
    shape, dim = (1, 2), 1
    v = torch.randn(shape, dtype=torch.bfloat16, device="npu:0")
    g = torch.randn([1 if i != dim else shape[i] for i in range(v.ndim)],
                    dtype=torch.bfloat16, device="npu:0")
    res = torch.ops.aten._weight_norm(v, g, dim).cpu().float()
    ref = torch.ops.aten._weight_norm(v.cpu().float(), g.cpu().float(), dim)
    assert torch.allclose(res, ref, rtol=1e-2, atol=1e-2), "weight_norm 偏差"


check("weight_norm pow(,3) (上游补丁独占保护)", _wn)

x2 = torch.randn(4, 8).to("npu:0")
y2 = torch.randn(4, 8).to("npu:0")
check("fused gelu_and_mul tanh", lambda: __import__(
    "flag_gems.fused.gelu_and_mul", fromlist=["gelu_and_mul"]).gelu_and_mul(x2, y2, approximate="tanh"))
check("fused geglu", lambda: __import__(
    "flag_gems.fused.geglu", fromlist=["geglu"]).geglu(x2, y2))

n_fail = sum(1 for _, c in results if not c)
print(f"\nTOTAL: {len(results)}, failed: {n_fail}")
sys.exit(1 if n_fail else 0)
