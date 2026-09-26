#!/usr/bin/env python3
"""Olaya identity: every logo, icon and favicon in the repo is generated here, from one geometry.

    uv run --with resvg-py --with pillow python packages/identity/build.py

Re-outline the type only when the wordmark or the social-card copy changes (fonts are not
committed; both are SIL OFL 1.1, from github.com/google/fonts):

    uv run --with uharfbuzz --with fonttools python packages/identity/build.py outline \\
        --serif InstrumentSerif-Regular.ttf --mono 'MartianMono[wdth,wght].ttf'

The mark: a light ring (the "o"), a bold logistic curve crossing it (the decision layer's
probability curve) and a coral point on the curve at the certified threshold, p = 0.85.
"""

import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# ---------------------------------------------------------------- palette
INDIGO, NIGHT, CORAL, CORAL_300, CHALK, MIST = "#25236B", "#16143D", "#D9573B", "#F08A70", "#FAFAF7", "#D9D8E6"
TONAL_RING = "#3A3870"  # the ring on a Night ground, where the full-contrast ring would compete

# ---------------------------------------------------------------- geometry (viewBox 0 0 120 120)
CX = CY = 60
RING_R, RING_W = 42, 4.5
CURVE_W = 10
X0, X1 = 8, 112            # the curve's flat tails run outside the ring
Y_LO, Y_HI = 86, 34        # p = 0 and p = 1
KT = 5.0                   # steepness, k times the half-width
TAU = 0.85                 # the operating point
DOT_R, HALO = 8, 2.5
# compact mark, for 48 px and below: no ring, heavier curve, larger point
C_X0, C_X1, C_W, C_DOT, C_HALO = 16, 104, 15, 13, 4


def _logistic(u):
    return 1 / (1 + math.exp(-KT * u))


def _p(u):  # normalised to exactly 0..1 over u in [-1, 1]
    return (_logistic(u) - _logistic(-1)) / (_logistic(1) - _logistic(-1))


def curve_d(x0=X0, x1=X1, n=24):
    """The logistic curve as a Catmull-Rom cubic path through n samples."""
    half = (X1 - X0) / 2
    pts = [(x, Y_LO - (Y_LO - Y_HI) * _p((x - CX) / half)) for x in (x0 + (x1 - x0) * i / n for i in range(n + 1))]
    d = "M%.2f %.2f" % pts[0]
    for i in range(n):
        a, b, c, e = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, n)]
        d += " C%.2f %.2f %.2f %.2f %.2f %.2f" % (
            b[0] + (c[0] - a[0]) / 6, b[1] + (c[1] - a[1]) / 6, c[0] - (e[0] - b[0]) / 6, c[1] - (e[1] - b[1]) / 6, c[0], c[1])
    return d


def point():
    half = (X1 - X0) / 2
    target = _logistic(-1) + TAU * (_logistic(1) - _logistic(-1))
    u = -math.log(1 / target - 1) / KT
    return CX + u * half, Y_LO - (Y_LO - Y_HI) * TAU


def symbol(ring, curve, dot, uid, compact=False, ox=0.0, oy=0.0, scale=1.0):
    """The mark as SVG elements. The halo is a mask, so the gap is real on any background."""
    px, py = point()
    r, halo = (C_DOT, C_HALO) if compact else (DOT_R, HALO)
    t = f' transform="translate({ox:g} {oy:g}) scale({scale:g})"' if (ox or oy or scale != 1) else ""
    body = (f'<path d="{curve_d(C_X0, C_X1) if compact else curve_d()}" fill="none" stroke="{curve}" '
            f'stroke-width="{C_W if compact else CURVE_W}" stroke-linecap="round"/>')
    if not compact:
        body = f'<circle cx="{CX}" cy="{CY}" r="{RING_R}" fill="none" stroke="{ring}" stroke-width="{RING_W}"/>' + body
    return (f'<g{t}><mask id="{uid}" maskUnits="userSpaceOnUse" x="-10" y="-10" width="140" height="140">'
            f'<rect x="-10" y="-10" width="140" height="140" fill="#fff"/><circle cx="{px:.2f}" cy="{py:.2f}" r="{r + halo}" fill="#000"/></mask>'
            f'<g mask="url(#{uid})">{body}</g><circle cx="{px:.2f}" cy="{py:.2f}" r="{r}" fill="{dot}"/></g>')


def svg(w, h, inner, title="Olaya"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}" height="{h:g}" viewBox="0 0 {w:g} {h:g}" role="img">'
            f'<title>{title}</title>{inner}</svg>\n')


# ---------------------------------------------------------------- type (outlined, see `outline`)
TYPE = os.path.join(HERE, "type.json")


def outline(argv):
    import uharfbuzz as hb
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    args = dict(zip(argv[::2], argv[1::2]))
    texts = {
        "wordmark": (args["--serif"], "Olaya"),
        "tagline1": (args["--serif"], "The coding harness"),
        "tagline2": (args["--serif"], "you can leave alone."),
        "repo": (args["--mono"], "open source · local decision layer · github.com/codelikeabhi/olaya"),
    }
    out = {}
    for key, (path, text) in texts.items():
        font = TTFont(path)
        upem = font["head"].unitsPerEm
        blob = hb.Blob.from_file_path(path)
        hbfont = hb.Font(hb.Face(blob))
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(hbfont, buf, {"kern": True, "liga": True})
        glyphset = font.getGlyphSet()
        order = font.getGlyphOrder()
        pen = SVGPathPen(glyphset, lambda v: "%.1f" % v)
        x = 0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            glyphset[order[info.codepoint]].draw(TransformPen(pen, (1, 0, 0, -1, x + pos.x_offset, -pos.y_offset)))
            x += pos.x_advance
        cap = font["OS/2"].sCapHeight if hasattr(font["OS/2"], "sCapHeight") else int(upem * 0.7)
        out[key] = {"upem": upem, "advance": x, "cap": cap, "d": pen.getCommands()}
    with open(TYPE, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", TYPE)


def text(key, size, fill, x, baseline):
    t = json.load(open(TYPE))[key]
    s = size / t["upem"]
    return f'<path transform="translate({x:.2f} {baseline:.2f}) scale({s:.5f})" fill="{fill}" d="{t["d"]}"/>', t["advance"] * s, t["cap"] * s


# ---------------------------------------------------------------- lockups
def _ybounds(key, size):
    """Top and bottom of an outlined text, relative to its baseline, at a font size."""
    import re
    t = json.load(open(TYPE))[key]
    ys, arity = [], {"M": 2, "L": 2, "T": 2, "Q": 4, "S": 4, "C": 6, "H": 1, "V": 1, "Z": 0}
    for cmd, args in re.findall(r"([MLTQSCHVZ])([^MLTQSCHVZ]*)", t["d"]):
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", args)]
        n = arity[cmd]
        for i in range(0, len(nums), n or 1):
            chunk = nums[i:i + n]
            ys += chunk if cmd == "V" else [] if cmd == "H" else chunk[1::2]
    return min(ys) * size / t["upem"], max(ys) * size / t["upem"]


def lockup(ink, ring, dot, uid):
    """Horizontal: symbol 120 high; the wordmark's cap height is centred on the ring's centre."""
    size = 112
    probe = json.load(open(TYPE))["wordmark"]
    cap = probe["cap"] * size / probe["upem"]
    baseline = CY + cap / 2
    top, bottom = _ybounds("wordmark", size)
    y0, y1 = min(0, baseline + top - 2), max(120, baseline + bottom + 2)
    word, adv, _ = text("wordmark", size, ink, 141, baseline - y0)
    return 141 + adv + 2, y1 - y0, symbol(ring, ink, dot, uid, oy=-y0) + word


def stacked(ink, ring, dot, uid):
    size = 96
    probe = json.load(open(TYPE))["wordmark"]
    adv = probe["advance"] * size / probe["upem"]
    cap = probe["cap"] * size / probe["upem"]
    _, bottom = _ybounds("wordmark", size)
    w = max(adv, 120) + 8
    baseline = 120 + 14 + cap
    word, _, _ = text("wordmark", size, ink, (w - adv) / 2, baseline)
    return w, baseline + bottom + 2, symbol(ring, ink, dot, uid, ox=(w - 120) / 2) + word


# ---------------------------------------------------------------- icons
CHANNELS = {  # tile, ring, curve, point
    "prod": (NIGHT, TONAL_RING, CHALK, CORAL_300),
    "beta": (CORAL, "#C24A30", CHALK, NIGHT),
    "dev": (CHALK, MIST, INDIGO, CORAL),
}


def tile_svg(size, colours, shape="rounded", inset=0.0, shadow=False, mark_scale=None, transparent=False):
    """An app icon: a tile (or none) with the mark centred. Small sizes get the compact mark."""
    tile, ring, curve, dot = colours
    compact = size <= 48
    s = size
    t0 = s * inset
    tw = s - 2 * t0
    parts = []
    if shadow:
        parts.append(f'<defs><filter id="sh" x="-20%" y="-20%" width="140%" height="140%">'
                     f'<feGaussianBlur in="SourceAlpha" stdDeviation="{s * 0.012:.2f}"/><feOffset dy="{s * 0.01:.2f}"/>'
                     f'<feComponentTransfer><feFuncA type="linear" slope="0.35"/></feComponentTransfer>'
                     f'<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
    if not transparent:
        attrs = f' filter="url(#sh)"' if shadow else ""
        if shape == "rounded":
            parts.append(f'<rect x="{t0:.2f}" y="{t0:.2f}" width="{tw:.2f}" height="{tw:.2f}" rx="{tw * 0.2237:.2f}" fill="{tile}"{attrs}/>')
        elif shape == "circle":
            parts.append(f'<circle cx="{s / 2}" cy="{s / 2}" r="{tw / 2:.2f}" fill="{tile}"{attrs}/>')
        else:
            parts.append(f'<rect width="{s}" height="{s}" fill="{tile}"/>')
    k = mark_scale if mark_scale is not None else (0.74 if compact else 0.78)
    m = tw * k
    parts.append(symbol(ring, curve, dot, "m", compact=compact, ox=t0 + (tw - m) / 2, oy=t0 + (tw - m) / 2, scale=m / 120))
    return svg(s, s, "".join(parts))


def render(svg_text, size):
    import resvg_py
    from PIL import Image
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg_text, width=size, height=size))
    return Image.open(io.BytesIO(png)).convert("RGBA")


def save_png(svg_text, size, path):
    render(svg_text, size).save(path, optimize=True)


def build_desktop(channel):
    """Re-render every icon the desktop build copies, keeping each file's size and convention."""
    root = os.path.join(REPO, "packages/desktop/icons", channel)
    colours = CHANNELS[channel]
    for dirpath, _, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            if not name.endswith(".png"):
                continue
            from PIL import Image
            size = Image.open(path).size[0]
            if rel.startswith("ios/"):
                doc = tile_svg(size, colours, shape="square")
            elif name == "ic_launcher_round.png":
                doc = tile_svg(size, colours, shape="circle", inset=1 / 24)
            elif name == "ic_launcher_foreground.png":
                doc = tile_svg(size, colours, transparent=True, mark_scale=0.5)
            elif name == "ic_launcher.png":
                doc = tile_svg(size, colours, inset=1 / 12)
            elif name == "dock.png":
                doc = tile_svg(size, colours, inset=0.0977, shadow=True)
            else:
                doc = tile_svg(size, colours)
            save_png(doc, size, path)
    # macOS: Apple's grid (an 824/1024 tile with a shadow), assembled by iconutil
    tmp = tempfile.mkdtemp()
    iconset = os.path.join(tmp, "icon.iconset")
    os.makedirs(iconset)
    for base in (16, 32, 128, 256, 512):
        for mult, suffix in ((1, ""), (2, "@2x")):
            px = base * mult
            save_png(tile_svg(px, colours, inset=0.0977, shadow=px >= 64), px, os.path.join(iconset, f"icon_{base}x{base}{suffix}.png"))
    subprocess.run(["iconutil", "-c", "icns", "-o", os.path.join(root, "icon.icns"), iconset], check=True)
    shutil.rmtree(tmp)
    ico_sizes = [16, 24, 32, 48, 64, 256]
    render(tile_svg(256, colours), 256).save(os.path.join(root, "icon.ico"), sizes=[(s, s) for s in ico_sizes])
    values = os.path.join(root, "android/values/ic_launcher_background.xml")
    if os.path.exists(values):
        with open(values, "w") as f:
            f.write(f'<?xml version="1.0" encoding="utf-8"?>\n<resources>\n  <color name="ic_launcher_background">{colours[0]}</color>\n</resources>')


def build_web():
    fav = os.path.join(REPO, "packages/ui/src/assets/favicon")
    prod = CHANNELS["prod"]
    icon = tile_svg(32, prod)
    for name in ("favicon.svg", "favicon-v3.svg"):
        open(os.path.join(fav, name), "w").write(icon)
    for name in ("favicon.ico", "favicon-v3.ico"):
        render(tile_svg(48, prod), 48).save(os.path.join(fav, name), sizes=[(16, 16), (32, 32), (48, 48)])
    for name in ("favicon-96x96.png", "favicon-96x96-v3.png"):
        save_png(tile_svg(96, prod), 96, os.path.join(fav, name))
    for name in ("apple-touch-icon.png", "apple-touch-icon-v3.png"):
        save_png(tile_svg(180, prod, shape="square"), 180, os.path.join(fav, name))
    for size in (192, 512):  # maskable: the mark stays inside the 80% safe circle
        save_png(tile_svg(size, prod, shape="square", mark_scale=0.62), size, os.path.join(fav, f"web-app-manifest-{size}x{size}.png"))
    manifest = os.path.join(fav, "site.webmanifest")
    m = json.load(open(manifest))
    m["theme_color"] = m["background_color"] = NIGHT
    with open(manifest, "w") as f:
        json.dump(m, f, indent=2)
        f.write("\n")


def build_social():
    """GitHub social preview, 1280 x 640."""
    w, h = 1280, 640
    word, _, _ = text("wordmark", 124, CHALK, 88, 190)
    t1, _, _ = text("tagline1", 58, CHALK, 90, 420)
    t2, _, _ = text("tagline2", 58, CHALK, 90, 486)
    repo, _, _ = text("repo", 19, "#A3A1C4", 92, 562)
    mark = symbol(TONAL_RING, CHALK, CORAL_300, "s", ox=820, oy=150, scale=340 / 120)
    doc = svg(w, h, f'<rect width="{w}" height="{h}" fill="{NIGHT}"/>{word}{t1}{t2}{repo}{mark}')
    render_to = os.path.join(REPO, "packages/ui/src/assets/images/social-share.png")
    import resvg_py
    from PIL import Image
    Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=doc)))).convert("RGB").save(render_to, optimize=True)


def build_identity():
    out = lambda name, content: open(os.path.join(HERE, name), "w").write(content)
    for name, (ink, ring, dot) in {"olaya-logo.svg": (INDIGO, INDIGO, CORAL), "olaya-logo-dark.svg": (CHALK, CHALK, CORAL_300)}.items():
        w, h, inner = lockup(ink, ring, dot, "h")
        out(name, svg(w, h, inner))
    for name, (ink, ring, dot) in {"olaya-logo-stacked.svg": (INDIGO, INDIGO, CORAL), "olaya-logo-stacked-dark.svg": (CHALK, CHALK, CORAL_300)}.items():
        w, h, inner = stacked(ink, ring, dot, "h")
        out(name, svg(w, h, inner))
    out("olaya-symbol.svg", svg(120, 120, symbol(INDIGO, INDIGO, CORAL, "h")))
    out("olaya-symbol-dark.svg", svg(120, 120, symbol(CHALK, CHALK, CORAL_300, "h")))
    out("olaya-app-icon.svg", tile_svg(512, CHANNELS["prod"]))
    # VS Code toolbar buttons (16 px, no tile): mark.svg on dark themes, mark-light.svg on light
    out("mark.svg", svg(120, 120, symbol("none", CHALK, CORAL_300, "h", compact=True)))
    out("mark-light.svg", svg(120, 120, symbol("none", INDIGO, CORAL, "h", compact=True)))
    for size in (96, 192, 512):
        save_png(tile_svg(size, CHANNELS["prod"]), size, os.path.join(HERE, f"mark-{size}x{size}.png"))
    save_png(tile_svg(512, CHANNELS["dev"]), 512, os.path.join(HERE, "mark-512x512-light.png"))


def build_ui():
    """The mark's geometry for the app's Solid components, so the UI and the files never drift."""
    size = 112
    probe = json.load(open(TYPE))["wordmark"]
    cap = probe["cap"] * size / probe["upem"]
    baseline = CY + cap / 2
    top, bottom = _ybounds("wordmark", size)
    y0, y1 = min(0, baseline + top - 2), max(120, baseline + bottom + 2)
    px, py = point()
    geo = {
        "ring": {"cx": CX, "cy": CY, "r": RING_R, "width": RING_W},
        "curve": {"d": curve_d(), "width": CURVE_W},
        "point": {"cx": round(px, 2), "cy": round(py, 2), "r": DOT_R, "halo": HALO},
        "wordmark": {"d": probe["d"], "x": 141, "y": round(baseline - y0, 3), "scale": size / probe["upem"],
                     "width": round(141 + probe["advance"] * size / probe["upem"] + 2, 3), "height": round(y1 - y0, 3),
                     "symbolY": round(-y0, 3)},
        "coral": CORAL,
    }
    out = os.path.join(REPO, "packages/ui/src/components/logo-geometry.ts")
    with open(out, "w") as f:
        f.write("// Generated by packages/identity/build.py from the identity geometry. Do not edit by hand.\n")
        f.write("export const geometry = " + json.dumps(geo, indent=2) + " as const\n")
    # match the repo's formatting; skipped quietly where bun is not installed
    subprocess.run(["bun", "x", "prettier", "--write", out], cwd=REPO, capture_output=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["outline"]:
        outline(sys.argv[2:])
        sys.exit(0)
    build_identity()
    build_ui()
    build_web()
    build_social()
    for channel in CHANNELS:
        build_desktop(channel)
    print("identity assets written")
