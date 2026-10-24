"""Extract Inter4K/60fps/UHD/* from Inter4K.zip with CRC verification.
Info-ZIP unzip rejects this Zip64 archive as an 'overlapped components' zip bomb
(false positive: no entries overlap), so use Python's zipfile instead.
Resumable: entries whose extracted size already matches are re-verified by CRC, then skipped."""
import sys, os, zipfile, zlib
src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
ents = [i for i in z.infolist() if i.filename.startswith("Inter4K/60fps/UHD/") and not i.is_dir()]
done = bad = 0
for i in ents:
    out = os.path.join(dst, i.filename)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) == i.file_size:
        c = 0
        with open(out, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                c = zlib.crc32(blk, c)
        if c == i.CRC:
            done += 1; continue
    tmp = out + ".part"
    with z.open(i) as r, open(tmp, "wb") as w:   # zipfile raises BadZipFile on CRC mismatch
        for blk in iter(lambda: r.read(1 << 24), b""):
            w.write(blk)
    os.replace(tmp, out); done += 1
print(f"{done}/{len(ents)} entries extracted and CRC-verified")
