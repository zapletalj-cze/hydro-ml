"""Checks why weirs snap to 0 points: reads sfincs.inp grid extent, finds the
referenced weirfile, and reports coordinate ranges + first lines to a txt."""

import re
from pathlib import Path

MODEL_ROOT = Path(r"D:\90_PersonalFoldlers\JZa\DataProcessing\levees_detection\SFINCS_model\model_RP100\sfincs_levees")
OUT_TXT = Path(__file__).parent / "diagnostics_ch4" / "weir_check.txt"


def main():
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    w = lines.append

    inp = (MODEL_ROOT / "sfincs.inp").read_text()
    def get(key, cast=float):
        m = re.search(rf"^\s*{key}\s*=\s*(\S+)", inp, flags=re.M)
        return cast(m.group(1)) if m else None

    x0, y0 = get("x0"), get("y0")
    dx, dy = get("dx"), get("dy")
    mmax, nmax = get("mmax", int), get("nmax", int)
    rot = get("rotation") or 0.0
    xmax, ymax = x0 + mmax * dx, y0 + nmax * dy
    w(f"grid: x0={x0} y0={y0} dx={dx} dy={dy} mmax={mmax} nmax={nmax} "
      f"rotation={rot}")
    w(f"grid extent: x [{x0} .. {xmax}]  y [{y0} .. {ymax}]")

    m = re.search(r"^\s*weirfile\s*=\s*(\S+)", inp, flags=re.M)
    if not m:
        w("NO weirfile entry in sfincs.inp")
        OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
        print("written:", OUT_TXT)
        return
    weir_path = MODEL_ROOT / m.group(1)
    w(f"weirfile referenced: {m.group(1)} (exists: {weir_path.exists()})")

    raw = weir_path.read_text().splitlines()
    w(f"weirfile lines: {len(raw)}")
    w("--- first 8 lines ---")
    for ln in raw[:8]:
        w(ln)

    xs, ys, zs, n_blocks, bad = [], [], [], 0, 0
    for ln in raw:
        parts = ln.split()
        if len(parts) == 4:
            try:
                x, y, z, cd = (float(p) for p in parts)
                xs.append(x); ys.append(y); zs.append(z)
            except ValueError:
                bad += 1
        elif len(parts) == 1 and parts[0]:
            n_blocks += 1
    w(f"blocks (names): {n_blocks} | coordinate rows: {len(xs)} "
      f"| unparsable rows: {bad}")
    if xs:
        w(f"x range: {min(xs):.1f} .. {max(xs):.1f}")
        w(f"y range: {min(ys):.1f} .. {max(ys):.1f}")
        w(f"z range: {min(zs):.2f} .. {max(zs):.2f}")
        inside = sum(1 for x, y in zip(xs, ys)
                     if x0 <= x <= xmax and y0 <= y <= ymax)
        w(f"vertices inside grid extent: {inside} / {len(xs)}")
        swapped = sum(1 for x, y in zip(xs, ys)
                      if x0 <= y <= xmax and y0 <= x <= ymax)
        w(f"vertices inside if x/y SWAPPED: {swapped} / {len(xs)}")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("written:", OUT_TXT)


if __name__ == "__main__":
    main()