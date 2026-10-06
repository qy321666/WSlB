"""诊断 c10.dll 加载失败:解析 PE 导入表,逐个尝试加载依赖,定位真正失败的 DLL。"""
import ctypes
import os
import struct
import sys

TORCH_LIB = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")


def read_imports(path):
    """解析 PE 文件的导入表,返回依赖 DLL 名字列表。"""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"MZ":
        raise ValueError("not a PE file")
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_off:pe_off + 4] != b"PE\0\0":
        raise ValueError("bad PE signature")
    coff = pe_off + 4
    n_sections = struct.unpack_from("<H", data, coff + 2)[0]
    opt_size = struct.unpack_from("<H", data, coff + 16)[0]
    opt = coff + 20
    magic = struct.unpack_from("<H", data, opt)[0]
    is_pe32p = magic == 0x20B
    dd_off = opt + (112 if is_pe32p else 96)
    # 导入表是数据目录第 2 项 (index 1)
    imp_rva, imp_size = struct.unpack_from("<II", data, dd_off + 8)
    sections = []
    sec_off = opt + opt_size
    for i in range(n_sections):
        s = sec_off + i * 40
        vaddr, vsize = struct.unpack_from("<II", data, s + 12)
        raw_size, raw_ptr = struct.unpack_from("<II", data, s + 16)
        sections.append((vaddr, max(vsize, raw_size), raw_ptr))

    def rva2off(rva):
        for vaddr, size, raw_ptr in sections:
            if vaddr <= rva < vaddr + size:
                return raw_ptr + (rva - vaddr)
        return None

    off = rva2off(imp_rva)
    names = []
    while True:
        entry = struct.unpack_from("<IIIII", data, off)
        if all(v == 0 for v in entry):
            break
        name_rva = entry[3]
        no = rva2off(name_rva)
        if no is not None:
            end = data.index(b"\0", no)
            names.append(data[no:end].decode("ascii", "replace"))
        off += 20
    return names


def main():
    target = os.path.join(TORCH_LIB, "c10.dll")
    print(f"[file] {target}  exists={os.path.exists(target)}")
    print(f"[arch] python {struct.calcsize('P') * 8}-bit")
    imports = read_imports(target)
    print(f"[imports] c10.dll 共依赖 {len(imports)} 个 DLL:")
    for n in imports:
        print("   -", n)

    # 加 torch/lib 到 DLL 搜索路径(与 torch/__init__.py 的做法一致)
    try:
        os.add_dll_directory(TORCH_LIB)
    except Exception as e:
        print("[warn] add_dll_directory:", e)

    print("\n[load] 逐个尝试加载依赖:")
    for n in imports:
        try:
            ctypes.WinDLL(n)
            print(f"   OK    {n}")
        except OSError as e:
            print(f"   FAIL  {n}: {e}")

    print("\n[load] 直接加载 c10.dll:")
    try:
        ctypes.WinDLL(os.path.join(TORCH_LIB, "c10.dll"))
        print("   OK")
    except OSError as e:
        print(f"   FAIL: {e}")


if __name__ == "__main__":
    main()
