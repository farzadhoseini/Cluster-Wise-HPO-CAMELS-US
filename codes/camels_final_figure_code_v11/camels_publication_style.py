from __future__ import annotations
"""
Publication style for the final CAMELS-US figure set.

Design goals
------------
- keep the user-approved Javi Del Ser color palette
- improve human-eye readability for paper figures
- use Calibri-family typography
- be safer for black-and-white printing by using contrast and line styles,
  but do NOT add distracting hatches/striping to heatmaps or legends
"""
from pathlib import Path
from cycler import cycler
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

JAVI_PALETTE = ["#56ae6c", "#8960b3", "#b0923b", "#ba495b"]
EXTENDED_PALETTE = JAVI_PALETTE + ["#2f5597", "#7f7f7f", "#111111", "#c0504d", "#4d4d4d"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
BW_SAFE_CMAP = "cividis"
DIVERGING_CMAP = "RdYlBu_r"

MODEL_COLORS = {
    "Cluster-wise": JAVI_PALETTE[0],
    "Top10": JAVI_PALETTE[1],
    "Benchmark": JAVI_PALETTE[2],
    "Reference": JAVI_PALETTE[2],
}
MODEL_LINESTYLES = {"Cluster-wise": "-", "Top10": "--", "Benchmark": "-.", "Reference": ":"}
MODEL_MARKERS = {"Cluster-wise": "o", "Top10": "s", "Benchmark": "^", "Reference": "D"}
CLUSTER_COLORS = {1: JAVI_PALETTE[0], 2: JAVI_PALETTE[1], 3: JAVI_PALETTE[2], 4: JAVI_PALETTE[3], 5: "#2f5597", 6: "#7f7f7f"}


def apply_publication_style() -> None:
    plt.rc("font", family="sans-serif")
    plt.rcParams["font.sans-serif"] = ["Calibri", "Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"]
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.titlesize": 22,
        "axes.labelsize": 19,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "axes.linewidth": 1.35,
        "lines.linewidth": 2.4,
        "lines.markersize": 5.2,
        "patch.linewidth": 0.9,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.26,
        "axes.grid": True,
        "image.cmap": BW_SAFE_CMAP,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": cycler(color=EXTENDED_PALETTE) + cycler(linestyle=(LINESTYLES * 2)[:len(EXTENDED_PALETTE)]),
    })


def get_model_color(model: str, i: int = 0) -> str:
    return MODEL_COLORS.get(str(model), EXTENDED_PALETTE[i % len(EXTENDED_PALETTE)])


def get_model_linestyle(model: str, i: int = 0):
    return MODEL_LINESTYLES.get(str(model), LINESTYLES[i % len(LINESTYLES)])


def get_model_marker(model: str, i: int = 0) -> str:
    return MODEL_MARKERS.get(str(model), MARKERS[i % len(MARKERS)])


def get_cluster_palette(clusters=None):
    if clusters is None:
        clusters = [1, 2, 3, 4, 5, 6]
    return {int(c): CLUSTER_COLORS.get(int(c), EXTENDED_PALETTE[(int(c) - 1) % len(EXTENDED_PALETTE)]) for c in clusters}


def _is_colorbar_axis(ax) -> bool:
    try:
        return getattr(ax, "_colorbar", None) is not None
    except Exception:
        return False




def _is_heatmap_axis(ax) -> bool:
    try:
        if len(getattr(ax, "images", [])) > 0:
            return True
    except Exception:
        pass
    try:
        for coll in getattr(ax, "collections", []):
            cname = coll.__class__.__name__.lower()
            if "quadmesh" in cname:
                return True
    except Exception:
        pass
    return False

def polish_axis(ax) -> None:
    # Do not aggressively redraw colorbars, and keep heatmaps clean.
    if not _is_colorbar_axis(ax):
        try:
            if _is_heatmap_axis(ax):
                ax.grid(False)
            else:
                ax.grid(True, color="0.85", linewidth=0.7, alpha=0.55)
        except Exception:
            pass
    for spine in getattr(ax, "spines", {}).values():
        try:
            spine.set_linewidth(1.15)
            spine.set_color("0.15")
        except Exception:
            pass
    try:
        ax.tick_params(width=1.0, length=4.2, colors="0.10", labelsize=14.5)
    except Exception:
        pass

    # Harmonize titles and labels across figures for consistent A4 appearance.
    try:
        cur = float(ax.title.get_fontsize())
        ax.title.set_fontsize(min(max(cur, 18), 20))
        ax.title.set_fontweight("bold")
    except Exception:
        pass
    try:
        curx = float(ax.xaxis.label.get_fontsize())
        cury = float(ax.yaxis.label.get_fontsize())
        ax.xaxis.label.set_fontsize(min(max(curx, 16), 17))
        ax.yaxis.label.set_fontsize(min(max(cury, 16), 17))
    except Exception:
        pass

    # Improve line visibility, but do NOT force markers onto smooth ECDF lines.
    for i, line in enumerate(ax.get_lines()):
        try:
            if line.get_linewidth() < 2.0:
                line.set_linewidth(2.3)
            if line.get_marker() not in (None, "None", ""):
                if line.get_markersize() > 4.0:
                    line.set_markersize(3.0)
                line.set_markeredgecolor("black")
                line.set_markeredgewidth(0.35)
            if line.get_linestyle() in (None, "None", ""):
                line.set_linestyle(LINESTYLES[i % len(LINESTYLES)])
        except Exception:
            pass

    # Scatter points: give light edge, but do not touch heatmap meshes/colorbars.
    for coll in getattr(ax, "collections", []):
        cname = coll.__class__.__name__.lower()
        try:
            if "pathcollection" in cname:  # scatter
                coll.set_linewidth(0.28)
                coll.set_edgecolor("black")
            elif "quadmesh" in cname:  # heatmaps
                try:
                    coll.set_edgecolor("face")
                except Exception:
                    pass
                try:
                    coll.set_linewidth(0.0)
                except Exception:
                    pass
        except Exception:
            pass

    # Bar/box patches: no hatches, no heavy black borders.
    for patch in getattr(ax, "patches", []):
        try:
            patch.set_hatch(None)
        except Exception:
            pass
        try:
            patch.set_linewidth(min(max(patch.get_linewidth(), 0.45), 0.8))
            patch.set_edgecolor("0.20")
        except Exception:
            pass

    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.get_frame().set_alpha(0.96)
            leg.get_frame().set_edgecolor("0.70")
        except Exception:
            pass
        try:
            title = leg.get_title()
            title.set_fontsize(min(max(float(title.get_fontsize()), 13), 14))
            title.set_fontweight("bold")
        except Exception:
            pass
        # make legend patch handles consistent and clean
        try:
            handles = getattr(leg, "legendHandles", None) or getattr(leg, "legend_handles", None)
            if handles is not None:
                for h in handles:
                    if hasattr(h, "set_hatch"):
                        h.set_hatch(None)
                    if hasattr(h, "set_linewidth"):
                        h.set_linewidth(0.8)
        except Exception:
            pass


def polish_figure(fig) -> None:
    for ax in fig.axes:
        polish_axis(ax)
    st = getattr(fig, "_suptitle", None)
    if st is not None:
        try:
            st.set_fontsize(max(float(st.get_fontsize()), 24))
            st.set_fontweight("bold")
        except Exception:
            pass
    for txt in fig.texts:
        try:
            txt.set_fontfamily("sans-serif")
        except Exception:
            pass
    try:
        fig.align_labels()
    except Exception:
        pass


def save_publication_figure(fig, outpath, dpi: int = 300, close: bool = False, bbox_inches: str = "tight") -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    polish_figure(fig)
    try:
        fig.tight_layout()
    except Exception:
        pass
    suffix = outpath.suffix.lower()
    raster_path = outpath if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} else outpath.with_suffix(".png")
    pdf_path = outpath.with_suffix(".pdf")
    fig.savefig(raster_path, dpi=dpi, bbox_inches=bbox_inches, facecolor="white")
    fig.savefig(pdf_path, bbox_inches=bbox_inches, facecolor="white")
    if close:
        plt.close(fig)


def install_dual_savefig_patch(default_dpi: int = 300) -> None:
    """Patch Figure.savefig so legacy scripts produce a sibling PDF automatically."""
    if getattr(Figure.savefig, "_camels_dual_save_patch", False):
        return
    original = Figure.savefig

    def patched(self, fname, *args, **kwargs):
        path = Path(fname) if not hasattr(fname, "write") else None
        if path is not None:
            kwargs.setdefault("dpi", default_dpi)
            kwargs.setdefault("bbox_inches", "tight")
            kwargs.setdefault("facecolor", "white")
            try:
                polish_figure(self)
            except Exception:
                pass
        result = original(self, fname, *args, **kwargs)
        if path is not None and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            pdf_path = path.with_suffix(".pdf")
            if not pdf_path.exists():
                pdf_kwargs = dict(kwargs)
                pdf_kwargs.pop("dpi", None)
                try:
                    original(self, pdf_path, *args, **pdf_kwargs)
                except Exception as e:
                    print(f"[WARN] Could not save PDF sibling for {path.name}: {e}")
        return result

    patched._camels_dual_save_patch = True
    Figure.savefig = patched

# Backward-compatible aliases
apply_camels_style = apply_publication_style
save_figure = save_publication_figure
