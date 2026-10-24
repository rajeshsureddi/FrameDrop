"""Render every Markdown table of tools/README_src.md as styled SVG images (light + dark) and write README.md.

GitHub strips colour and style from Markdown tables, so each table becomes
  <picture> (dark/light SVG) + a collapsible copy of the original Markdown table.
Styling (reference dataviz palette):
  header band blue-550 with white text; zebra rows; rows naming our model ("Ours", "v4", ...) tinted blue with an
  accent bar; **bold** segments in bold accent blue; *italic* / whole-italic rows muted grey; first column left-aligned,
  others centred; long cells wrap. Text is drawn as vector paths, so the image is identical in every viewer.

    python tools/render_tables.py        (from the repo root)
"""
import os
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402

plt.rcParams["svg.fonttype"] = "path"

SRC, OUT_MD, FIG = "tools/README_src.md", "README.md", "figures/tables"
THEMES = {
    "light": dict(surface="#fcfcfb", header="#1c5cab", header_ink="#ffffff", ink="#0b0b0b", muted="#898781",
                  zebra="#f2f1ee", ours="#e3eefc", accent="#2a78d6", best="#1c5cab", rule="#d9d8d3"),
    "dark": dict(surface="#1a1a19", header="#256abf", header_ink="#ffffff", ink="#f0efec", muted="#9a9890",
                 zebra="#222220", ours="#1b2a40", accent="#3987e5", best="#86b6ef", rule="#383835"),
}
FS = 10.5                       # font size in pt (drawn at dpi 100: ~14.6 px)
PAD_X, PAD_Y, LINE = 14, 9, 20  # cell padding and line height, px
MAX_COL = 330                   # px before a cell wraps
OURS = re.compile(r"\b[Oo]urs\b|^\*\*")    # rows naming our model, or with a bold first cell

FONT = {(b, i): FontProperties(family="DejaVu Sans", weight="bold" if b else "normal", style="italic" if i else "normal",
                                size=FS) for b in (0, 1) for i in (0, 1)}
_T2P = TextToPath()


def width(text, bold=False, italic=False):
    """advance width in px (dpi 100, size FS): what the next glyph is placed after"""
    if not text:
        return 0.0
    return _T2P.get_text_width_height_descent(text, FONT[(bool(bold), bool(italic))], ismath=False)[0] * 100 / 72  # pt -> px


SPACE = 0.85 * (width("a a") - width("aa"))   # slightly tighter than the font default


def segments(cell):
    """markdown cell -> [(text, bold, italic)]; supports **b**, *i*, `code`, [text](link)."""
    cell = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cell).replace("`", "").replace("<br/>", " ")
    out, pos = [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*|(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", cell):
        if m.start() > pos:
            out.append((cell[pos:m.start()], False, False))
        out.append((m.group(1), True, False) if m.group(1) is not None else (m.group(2), False, True))
        pos = m.end()
    if pos < len(cell):
        out.append((cell[pos:], False, False))
    return [s for s in out if s[0]] or [("", False, False)]


def wrap(segs, maxw):
    """greedy word wrap of styled segments -> list of lines, each a list of segments"""
    words = []
    for t, b, i in segs:
        for k, w in enumerate(re.split(r"(\s+)", t)):
            if w:
                words.append((w, b, i))
    lines, cur, cw = [], [], 0.0
    for w, b, i in words:
        ww = width(w, b, i) if not w.isspace() else SPACE
        if cur and not w.isspace() and cw + ww > maxw:
            while cur and cur[-1][0].isspace():
                cur.pop()
            lines.append(cur)
            cur, cw = [], 0.0
        if w.isspace() and not cur:
            continue
        cur.append((w, b, i))
        cw += ww
    if cur:
        lines.append(cur)
    return lines or [[("", False, False)]]


def line_width(line):
    return sum(width(w, b, i) if not w.isspace() else SPACE for w, b, i in line)


def parse(block):
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block]
    head, body = rows[0], [r for r in rows[2:]]
    n = len(head)
    body = [r + [""] * (n - len(r)) for r in body]
    return head, body


def render(head, body, path, th):
    n = len(head)
    cells = [[segments(c) for c in head]] + [[segments(c) for c in r] for r in body]
    natural = [max(line_width([(t, b or k == 0, i) for t, b, i in cells[k][j]]) for k in range(len(cells))) for j in range(n)]
    colw = [min(w, MAX_COL) + 2 * PAD_X for w in natural]
    wrapped = [[wrap([(t, b or k == 0, i) for t, b, i in cells[k][j]], colw[j] - 2 * PAD_X) for j in range(n)]
               for k in range(len(cells))]
    rowh = [max(len(c) for c in r) * LINE + 2 * PAD_Y for r in wrapped]

    def wordy(j):                                       # mostly letters -> left-align; numbers stay centred
        txt = "".join(r[j] for r in body)
        letters = sum(ch.isalpha() for ch in txt)
        return j == 0 or letters > 0.5 * max(1, sum(ch.isalnum() for ch in txt))
    left = [wordy(j) for j in range(n)]
    W, H = sum(colw) + 6, sum(rowh) + 2
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor(th["surface"])
    ax.add_patch(Rectangle((0, 0), W, H, facecolor=th["surface"], edgecolor="none"))
    y = 1
    for k, r in enumerate(wrapped):
        raw = " ".join(head if k == 0 else body[k - 1])
        is_head = k == 0
        first = (head if is_head else body[k - 1])[0]
        ours = not is_head and bool(OURS.search(first))
        muted = not is_head and raw.strip().startswith("*") and not raw.strip().startswith("**")
        if is_head:
            ax.add_patch(FancyBboxPatch((1, y), W - 2, rowh[k], boxstyle="round,pad=0,rounding_size=6",
                                            facecolor=th["header"], edgecolor="none"))
        elif ours:
            ax.add_patch(Rectangle((1, y), W - 2, rowh[k], facecolor=th["ours"], edgecolor="none"))
            ax.add_patch(Rectangle((1, y), 4, rowh[k], facecolor=th["accent"], edgecolor="none"))
        elif k % 2 == 0:
            ax.add_patch(Rectangle((1, y), W - 2, rowh[k], facecolor=th["zebra"], edgecolor="none"))
        if not is_head:
            ax.plot([1, W - 1], [y + rowh[k], y + rowh[k]], color=th["rule"], lw=0.6)
        x = 3
        for j, lines in enumerate(r):
            ty = y + PAD_Y + (rowh[k] - 2 * PAD_Y - len(lines) * LINE) / 2 + LINE * 0.72
            for line in lines:
                lw = line_width(line)
                cx = x + PAD_X + (6 if j == 0 and ours else 0) if left[j] else x + (colw[j] - lw) / 2
                for w, b, i in line:
                    if w.isspace():
                        cx += SPACE
                        continue
                    if is_head:
                        col = th["header_ink"]
                    elif muted or i:
                        col = th["muted"]
                    elif b and j > 0:
                        col = th["best"]
                    else:
                        col = th["ink"]
                    ax.text(cx, ty, w.replace("$", r"\$"), fontproperties=FONT[(b or is_head, i)], fontsize=FS,
                            color=col, va="baseline", ha="left")
                    cx += width(w, b or is_head, i)
                ty += LINE
            x += colw[j]
        y += rowh[k]
    fig.savefig(path, format="svg", facecolor=th["surface"])
    if os.environ.get("PREVIEW"):                        # PNG copies for a visual check (not committed)
        fig.savefig(os.path.join(os.environ["PREVIEW"], os.path.basename(path)[:-4] + ".png"), dpi=100,
                    facecolor=th["surface"])
    plt.close(fig)
    return W


def main():
    os.makedirs(FIG, exist_ok=True)
    for f in os.listdir(FIG):
        os.remove(os.path.join(FIG, f))
    lines = open(SRC).read().split("\n")
    out, i, n, title, in_code = [], 0, 0, "Table", False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{2,4} ", ln):
            title = re.sub(r"^#+ ", "", ln).strip()
        if not in_code and ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            block = lines[i:j]
            head, body = parse(block)
            n += 1
            name = f"t{n:02d}"
            w = 0
            for t, th in THEMES.items():
                w = render(head, body, f"{FIG}/{name}_{t}.svg", th)
            alt = f"{title}: table {n}"
            out += ["<picture>",
                    f'  <source media="(prefers-color-scheme: dark)" srcset="{FIG}/{name}_dark.svg">',
                    f'  <img alt="{alt}" src="{FIG}/{name}_light.svg" width="{min(int(w), 900)}">',
                    "</picture>", "", "<details><summary>table as text</summary>", ""] + block + ["", "</details>"]
            i = j
            continue
        out.append(ln)
        i += 1
    open(OUT_MD, "w").write("\n".join(out))
    print(f"{n} tables rendered -> {FIG}, {OUT_MD} written")


if __name__ == "__main__":
    main()
