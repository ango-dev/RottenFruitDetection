"""Chart style and image helpers shared by the notebooks."""

import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import StrMethodFormatter

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # series 1 and 2
CONTEXT = "#d5d3cb"  # bars that are shown for context only


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def render(fig):
    """Show a figure inline as a PNG, whatever the matplotlib backend is (Ultralytics switches it)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    display(Image(data=buf.getvalue(), format="png"))


def hbars(ax, labels, values, colors=BLUE, fmt="{:,}", tips=True, xmax=None, bar_px=16, radius_px=4):
    """Horizontal bars from a shared baseline: rounded data end, square base, value at the tip."""
    colors = [colors] * len(values) if isinstance(colors, str) else list(colors)
    n = len(values)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, xmax or max(values) * 1.15)
    ax.set_yticks(range(n), labels)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.figure.canvas.draw()  # final layout, so pixel sizes below are right
    box = ax.get_window_extent()
    xpp = (ax.get_xlim()[1] - ax.get_xlim()[0]) / box.width  # data units per pixel
    ypp = (ax.get_ylim()[0] - ax.get_ylim()[1]) / box.height
    h, r = bar_px * ypp, radius_px * xpp
    for i, (v, c) in enumerate(zip(values, colors)):
        rr = min(r, v / 2)
        ax.add_patch(FancyBboxPatch((0, i - h / 2), v, h, boxstyle=f"round,pad=0,rounding_size={rr}",
                                    mutation_aspect=ypp / xpp, fc=c, ec="none", zorder=2))
        ax.add_patch(Rectangle((0, i - h / 2), max(v - rr, 0), h, fc=c, ec="none", zorder=2))
        if tips:
            ax.text(v + 6 * xpp, i, fmt.format(v), va="center", ha="left", color=INK_2, fontsize=9)


def hex_bgr(color):
    return tuple(int(color[i:i + 2], 16) for i in (5, 3, 1))


def draw_box(img, xyxy, color=BLUE, label=None):
    """Draw a pixel xyxy box with an optional filled label tag (in place), sized to stay
    readable after grid() shrinks the image to a tile."""
    s = img.shape[1] / 300
    x1, y1, x2, y2 = map(int, xyxy)
    bgr, fs, pad = hex_bgr(color), 0.55 * s, int(5 * s)
    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, max(2, round(2.5 * s)), cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, round(s)))
        ty = max(y1, th + 2 * pad)
        cv2.rectangle(img, (x1, ty - th - 2 * pad), (x1 + tw + 2 * pad, ty), bgr, -1)
        cv2.putText(img, label, (x1 + pad, ty - pad), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                    max(1, round(s)), cv2.LINE_AA)
    return img


def yolo_to_xyxy(box, w, h):
    xc, yc, bw, bh = box
    return (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h


def show(path, width=900, fmt="jpeg"):
    """Display an image file (e.g. an Ultralytics plot) inline, scaled down to `width`."""
    img = cv2.imread(str(path))
    if img.shape[1] > width:
        img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(f".{'jpg' if fmt == 'jpeg' else fmt}", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    display(Image(data=buf.tobytes(), format=fmt))


def grid(images, captions=None, cols=4, tile=240, gap=4):
    """Tile BGR images (with optional captions under each) and show them inline as one JPEG."""
    captions = captions or [None] * len(images)
    cells = []
    for im, cap in zip(images, captions):
        cell = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA)
        if cap:
            strip = np.full((26, tile, 3), hex_bgr(SURFACE), np.uint8)
            cv2.putText(strip, cap, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hex_bgr(INK_2), 1, cv2.LINE_AA)
            cell = np.vstack([cell, strip])
        cells.append(cell)
    blank = np.full_like(cells[0], hex_bgr(SURFACE))
    cells += [blank] * (-len(cells) % cols)
    vgap = np.full((cells[0].shape[0], gap, 3), hex_bgr(SURFACE), np.uint8)
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        rows.append(np.hstack([x for c in row for x in (c, vgap)][:-1]))
    hgap = np.full((gap, rows[0].shape[1], 3), hex_bgr(SURFACE), np.uint8)
    sheet = np.vstack([x for r in rows for x in (r, hgap)][:-1])
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    display(Image(data=buf.tobytes(), format="jpeg"))
