#!/usr/bin/env python3
"""安装/卸载/检查 cann libdevice pow int-指数提升补丁 (上游层修复)。

用法:
  python apply_upstream_patch.py status [triton_path]
  python apply_upstream_patch.py apply  [triton_path]   # 幂等
  python apply_upstream_patch.py revert [triton_path]   # 恢复 .orig

默认 triton_path 自动探测 (site-packages)。补丁内容与
libdevice_pow_promotion.patch 一致; apply 前自动备份 .orig。
"""
import importlib.util
import sys

MARK = "wt-2026-09-10-fix (FlagGems #5867)"

PATCH_BODY = '''    # wt-2026-09-10-fix (FlagGems #5867): a Python int literal exponent
    # (pow(x, 2)) reaches here as constexpr[2] via the JIT call path and
    # extern_elementwise's to_tensor types it int32, but the dispatch tables
    # only contain float/float pairs -> KeyError(float32, int32) at compile
    # time (the whole gelu-tanh family died on this). Promote integral
    # constexpr/bare-int exponents to float so the lookup lands on (fp,fp).
    # wt <wangt635@ustc.edu.cn>
    if type(arg1) is int:
        arg1 = float(arg1)
    elif type(arg1).__name__ == "constexpr":
        try:
            v = arg1.value
            if type(v) is int and getattr(arg0, "dtype", None) is not None:
                arg1 = float(v)
        except AttributeError:
            pass
'''

ANCHOR = '''@core.extern
def pow(arg0, arg1, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:'''

PATCHED_ANCHOR = '''@core.extern
def pow(arg0, arg1, _semantic=None):
''' + PATCH_BODY + '''    if triton_enable_libdevice_simt() and is_compile_on_910_95:'''


def find_libdevice():
    spec = importlib.util.find_spec("triton")
    if spec is None or spec.origin is None:
        sys.exit("triton not installed?")
    import os
    p = os.path.join(spec.origin.rsplit("triton/", 1)[0],
                     "triton", "language", "extra", "cann", "libdevice.py")
    return p


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    path = sys.argv[2] if len(sys.argv) > 2 else find_libdevice()
    src = open(path).read()

    if MARK in src:
        state = "PATCHED"
    elif ANCHOR in src:
        state = "ORIG"
    else:
        state = "UNKNOWN (pow 函数体与预期不符, 拒绝操作)"

    if action == "status":
        print(f"{path}\nstate: {state}")
        return 0 if state != "UNKNOWN" else 1

    if action == "apply":
        if state == "PATCHED":
            print("already patched, nothing to do")
            return 0
        if state != "ORIG":
            sys.exit(f"refuse: {state}")
        import shutil
        shutil.copy(path, path + ".orig")
        open(path, "w").write(src.replace(ANCHOR, PATCHED_ANCHOR))
        print(f"applied (backup: {path}.orig)")
        return 0

    if action == "revert":
        import os
        if os.path.exists(path + ".orig"):
            import shutil
            shutil.copy(path + ".orig", path)
            print("reverted from .orig")
            return 0
        if state == "ORIG":
            print("already original")
            return 0
        sys.exit("no .orig backup and not patchable state; manual fix needed")

    sys.exit(f"unknown action: {action}")


if __name__ == "__main__":
    sys.exit(main())
