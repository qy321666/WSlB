"""校验 torch 安装是否与 wheel 记录一致(RECORD 里的 sha256),并检查 PE 节区是否完整。"""
import base64
import csv
import hashlib
import os
import struct
import sys

SP = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
DIST = os.path.join(SP, "torch-2.11.0+cu128.dist-info", "RECORD")


def b64_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()


def check_pe(path):
    with open(path, "rb") as f:
        data = f.read()
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    coff = pe_off + 4
    n_sections = struct.unpack_from("<H", data, coff + 2)[0]
    opt_size = struct.unpack_from("<H", data, coff + 16)[0]
    sec_off = coff + 20 + opt_size
    worst = 0
    for i in range(n_sections):
        s = sec_off + i * 40
        name = data[s:s + 8].rstrip(b"\0").decode("ascii", "replace")
        raw_size, raw_ptr = struct.unpack_from("<II", data, s + 16)
        end = raw_ptr + raw_size
        worst = max(worst, end)
    return len(data), worst


def main():
    print(f"[RECORD] {DIST}  exists={os.path.exists(DIST)}")
    rows = list(csv.reader(open(DIST, encoding="utf-8")))
    bad, checked = [], 0
    interesting = ("torch/lib/c10.dll", "torch/lib/torch_cpu.dll", "torch/lib/libiomp5md.dll",
                   "torch/__init__.py", "torch/lib/c10_cuda.dll", "torch/lib/torch.dll")
    for row in rows:
        name, digest = row[0], row[1]
        if not digest:
            continue
        _, _, digest = digest.partition("=")
        if name not in interesting:
            continue
        p = os.path.join(SP, name.replace("/", os.sep))
        if not os.path.exists(p):
            bad.append((name, "MISSING"))
            continue
        actual = b64_sha256(p)
        ok = actual == digest
        checked += 1
        print(f"   {'OK  ' if ok else 'BAD '} {name}  ({os.path.getsize(p):,} bytes)")
        if not ok:
            bad.append((name, f"hash mismatch\n        recorded={digest}\n        actual  ={actual}"))

    print(f"\n[hash] 校验 {checked} 个关键文件, 失败 {len(bad)} 个")
    for n, why in bad:
        print(f"   {n}: {why}")

    print("\n[pe] 节区完整性检查:")
    for name in interesting:
        p = os.path.join(SP, name.replace("/", os.sep))
        if not os.path.exists(p) or not name.endswith(".dll"):
            continue
        size, worst = check_pe(p)
        flag = "OK" if worst <= size else "TRUNCATED!"
        print(f"   {flag} {name}: 文件 {size:,} 字节, 节区最远到 {worst:,} 字节")


if __name__ == "__main__":
    main()
