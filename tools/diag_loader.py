"""深入诊断:列出已加载模块、TLS 槽余量，并用多种 LoadLibrary 方式尝试 c10.dll。"""
import ctypes
import ctypes.wintypes as wt
import os
import struct
import sys

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)

TORCH_LIB = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")


def loaded_modules():
    h = k32.GetCurrentProcess()
    needed = wt.DWORD(0)
    psapi.EnumProcessModules(h, None, 0, ctypes.byref(needed))
    n = needed.value // ctypes.sizeof(wt.HMODULE)
    arr = (wt.HMODULE * n)()
    psapi.EnumProcessModules(h, arr, needed.value, ctypes.byref(needed))
    buf = ctypes.create_unicode_buffer(1024)
    out = []
    for i in range(n):
        psapi.GetModuleFileNameExW(h, arr[i], buf, 1024)
        out.append(buf.value)
    return out


def tls_free_slots():
    """不断分配 TLS 槽直到失败,统计可用数量(证明是否 TLS 耗尽)。"""
    k32.TlsAlloc.restype = wt.DWORD
    k32.TlsFree.argtypes = [wt.DWORD]
    slots = []
    while True:
        idx = k32.TlsAlloc()
        if idx == 0xFFFFFFFF:
            break
        slots.append(idx)
        if len(slots) > 2000:
            break
    for s in slots:
        k32.TlsFree(s)
    return len(slots)


def try_load(path, flags, label):
    k32.LoadLibraryExW.restype = wt.HMODULE
    k32.LoadLibraryExW.argtypes = [wt.LPCWSTR, wt.HANDLE, wt.DWORD]
    ctypes.set_last_error(0)
    h = k32.LoadLibraryExW(path, None, flags)
    err = ctypes.get_last_error()
    if h:
        print(f"   OK    {label}")
        k32.FreeLibrary(h)
    else:
        msg = ctypes.FormatError(err).strip()
        print(f"   FAIL  {label}: [{err}] {msg}")
    return bool(h)


def has_manifest(path):
    """粗略检查 PE 资源里是否含 RT_MANIFEST(类型 24)。"""
    with open(path, "rb") as f:
        data = f.read()
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    coff = pe_off + 4
    opt_size = struct.unpack_from("<H", data, coff + 16)[0]
    opt = coff + 20
    magic = struct.unpack_from("<H", data, opt)[0]
    dd_off = opt + (112 if magic == 0x20B else 96)
    res_rva = struct.unpack_from("<I", data, dd_off + 2 * 8)[0]
    return res_rva != 0


def main():
    mods = loaded_modules()
    print(f"[modules] 当前进程已加载 {len(mods)} 个 DLL")
    for m in mods:
        print("   ", m)

    print(f"\n[tls] 可用 TLS 槽: {tls_free_slots()}")

    p = os.path.join(TORCH_LIB, "c10.dll")
    print(f"\n[manifest] c10.dll 含资源节: {has_manifest(p)}")

    print("\n[load] 多种方式加载 c10.dll:")
    LOAD_WITH_ALTERED_SEARCH_PATH = 0x8
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x100
    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x1000
    LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x800

    ctypes.windll.kernel32.SetDllDirectoryW(None)
    try:
        os.add_dll_directory(TORCH_LIB)
    except Exception:
        pass
    try_load(p, 0, "LoadLibraryEx(flags=0, 全路径)")
    try_load(p, LOAD_WITH_ALTERED_SEARCH_PATH, "LOAD_WITH_ALTERED_SEARCH_PATH")
    try_load(p, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
             "SEARCH_DLL_LOAD_DIR|DEFAULT_DIRS")
    print("   短名 c10.dll:", end=" ")
    try:
        ctypes.WinDLL("c10.dll")
        print("OK")
    except OSError as e:
        print("FAIL", e)

    # 对比:同目录下其它 torch 自研 DLL 是否也失败
    print("\n[load] 对比同目录其它 DLL:")
    for name in ("libiomp5md.dll", "libiompstubs5md.dll", "c10_cuda.dll", "torch_cpu.dll"):
        q = os.path.join(TORCH_LIB, name)
        if os.path.exists(q):
            try_load(q, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, name)


if __name__ == "__main__":
    main()
