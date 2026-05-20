import os, struct, glob

root_files = glob.glob("/standard/ldmxuva/EaT_background_8gev/eat_note_run_2026/EaT_eot_equivalent_enriched/*.root")

print(f"{'File':<80} {'fEND':>12} {'Actual':>14} {'Status'}")
print("-" * 115)

for path in sorted(root_files):
    actual = os.path.getsize(path)
    with open(path, "rb") as f:
        header = f.read(100)
    version = struct.unpack(">I", header[4:8])[0]
    if version >= 1000000:
        fend = struct.unpack(">Q", header[41:49])[0]
    else:
        fend = struct.unpack(">I", header[33:37])[0]
    
    if fend <= 101:
        status = "❌ DEAD (fEND=header only)"
    elif fend > actual:
        status = "⚠️  TRUNCATED"
    else:
        status = "✅ OK"
    
    fname = os.path.basename(path)
    print(f"{fname:<80} {fend:>12} {actual:>14}  {status}")
