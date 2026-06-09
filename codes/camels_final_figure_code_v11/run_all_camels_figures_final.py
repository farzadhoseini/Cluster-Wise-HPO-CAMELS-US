from __future__ import annotations
"""
One-run final figure builder for the CAMELS-US paper.

Key behavior in this V11 update
-------------------------------
- keeps the final deliverable folder clean: only the final figure PDFs are left in
  <project-root>/results/All_plots/.
- all intermediate figure outputs are written to a temporary build folder and are
  deleted at the end of the run.
- final outputs follow the agreed Excel/reference figure set only.
"""
import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from camels_publication_style import (
    apply_publication_style,
    install_dual_savefig_patch,
    save_publication_figure,
    CLUSTER_COLORS,
    JAVI_PALETTE,
)

FINAL_SOURCE_MAP = {
    "fig01_map_clusters.png": "CAMELS_US_hydrology/fig01_map_clusters.png",
    "fig02_hydrological_signature_distributions.png": "CAMELS_US_hydrology/fig02_hydrological_signature_distributions.png",
    "fig03_cluster_profile_heatmap.png": "CAMELS_US_hydrology/fig04_cluster_profile_heatmap.png",
    "fig04_BOX_basins__NSE.png": "Results/plot_01_hydrograph_overview_v04/NSE/BOX_basins__NSE.png",
    "fig05_ECDF_split_basins__NSE.png": "Results/plot_01_hydrograph_overview_v04/NSE/ECDF_split_basins__NSE.png",
    "fig06_Trend_water_year__NSE.png": "Results/plot_02_water_year_v03/NSE/Trend_water_year__NSE.png",
    "fig07_Heatmap_year_x_model__NSE.png": "Results/plot_02_water_year_v03/NSE/Heatmap_year_x_model__NSE.png",
    "fig09_Heatmap_CLUSTERWISE_ONLY__NSE.png": "Results/plot_03_seasonal_v03/NSE/Heatmap_CLUSTERWISE_ONLY__NSE.png",
    "fig10_cluster_metric_heatmap_2x2_NSE.png": "Results/Tables_cluster/fig_cluster_metric_heatmap_2x2_NSE.png",
    "fig11_bad_basins_clusterwise_nse_lt_0p5_map.png": "cluster_performance_outputs/fig_bad_basins_clusterwise_nse_lt_0p5_map.png",
    "fig12_clusterwise_geo_nse_journal.png": "Results/Tables_cluster/fig_clusterwise_geo_nse_journal.png",
    "fig13_climate_topography_distributions.png": "CAMELS_US_hydrology/fig03_climate_topography_distributions.png",
    "fig14_pca_cluster_attribute_space.png": "CAMELS_US_hydrology/fig05_pca_cluster_attribute_space.png",
    "fig15_cluster_huc02_composition.png": "CAMELS_US_hydrology/fig06_cluster_huc02_composition.png",
    "fig16_nse_lt_0p5_effect_sizes.png": "Results/deep_diagnostic_camels_nse/nse_lt_0p5_effect_sizes.png",
    "fig17_water_year_nse_lt_0p5_counts_by_cluster.png": "Results/deep_diagnostic_camels_nse/water_year_nse_lt_0p5_counts_by_cluster.png",
    "fig18_weak_year_effect_sizes.png": "Results/deep_diagnostic_camels_nse/weak_year_effect_sizes.png",
    "fig19_seasonal_nse_lt_0p5_counts_by_cluster.png": "Results/deep_diagnostic_camels_nse/seasonal_nse_lt_0p5_counts_by_cluster.png",
    "fig20_jas_lt_0p5_effect_sizes.png": "Results/deep_diagnostic_camels_nse/jas_lt_0p5_effect_sizes.png",
    "fig21_seasonality_link_effect_sizes.png": "Results/deep_diagnostic_camels_nse/seasonality_link_effect_sizes.png",
    "fig22_water_year_median_nse_ranking.png": "Results/deep_diagnostic_camels_nse/water_year_median_nse_ranking.png",
    "fig23_seed_robustness_box_nse.png": "Results/plot_01_hydrograph_overview_v04/NSE/BOX_seed_robustness__NSE.png",
    "fig24_seed_robustness_cdf_nse.png": "Results/plot_01_hydrograph_overview_v04/NSE/CDF_seed_robustness__NSE.png",
    "fig25_wy1992_seed_box_nse.png": "Results/plot_02_water_year_v03/NSE/BOX_seeds_WY1992__NSE.png",
    "fig26_wy1992_seed_cdf_nse.png": "Results/plot_02_water_year_v03/NSE/CDF_seeds_WY1992__NSE.png",
    "fig27_wy1998_seed_box_nse.png": "Results/plot_02_water_year_v03/NSE/BOX_seeds_WY1998__NSE.png",
    "fig28_wy1998_seed_cdf_nse.png": "Results/plot_02_water_year_v03/NSE/CDF_seeds_WY1998__NSE.png",
}


def parse_args():
    p = argparse.ArgumentParser(description="Run all final CAMELS-US paper figures and collect PDF figures into results/All_plots.")
    p.add_argument("--project-root", required=True, help="CAMELS-US project root folder")
    p.add_argument("--outdir", default="results/All_plots", help="Final collection folder relative to project root.")
    p.add_argument("--metrics", default="NSE")
    p.add_argument("--models", default="Benchmark,Cluster-wise,Top10")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--clean-outdir", action="store_true")
    p.add_argument("--skip-deep-diagnostics", action="store_true")
    p.add_argument("--skip-pro-cluster-map", action="store_true")
    p.add_argument("--seed-years", default="1992,1998")
    p.add_argument("--strict-final-folder", action="store_true", default=True)
    return p.parse_args()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def rel_to_project(path: Path, project_root: Path) -> str:
    """Return a portable relative path from project_root to path."""
    try:
        return os.path.relpath(path.resolve(), project_root.resolve()).replace("\\", "/")
    except Exception:
        return str(path)


def rel_to_project_results(path: Path, project_root: Path) -> str:
    """Return relative path from <project_root>/Results to path.

    Some legacy scripts always do project_root / 'Results' / output_subdir.
    This function lets us still send their outputs to All_plots/others.
    """
    base = (project_root / "Results").resolve()
    try:
        return os.path.relpath(path.resolve(), base).replace("\\", "/")
    except Exception:
        return str(path)


def run_module_main(module_name: str, argv: list[str] | None = None, patch=None) -> None:
    apply_publication_style()
    old_argv = sys.argv[:]
    try:
        sys.argv = [module_name + ".py"] + (argv or [])
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
        if patch is not None:
            patch(module)
        module.main()
    finally:
        sys.argv = old_argv


def patch_camels_utils(build_root: Path):
    import camels_utils

    def _get_results_dir(script_name: str) -> Path:
        out = build_root / "Results" / script_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    camels_utils.get_results_dir = _get_results_dir
    return camels_utils


def patch_cluster_modules(module) -> None:
    if hasattr(module, "CLUSTER_COLORS"):
        module.CLUSTER_COLORS = dict(CLUSTER_COLORS)


def patch_bad_basins_module(project_root: Path, build_root: Path):
    def _patch(module):
        module.PROJECT_ROOT = project_root
        module.ATTR_FILE = build_root / "CAMELS_US_hydrology" / "camels_531_master_attributes_clusters.csv"
        if not module.ATTR_FILE.exists():
            alt = project_root / "CAMELS_US_hydrology" / "camels_531_master_attributes_clusters.csv"
            if alt.exists():
                module.ATTR_FILE = alt
        module.BAD_FILE = build_root / "Results" / "Tables_cluster" / "bad_basins_clusterwise_nse_lt_0p5.csv"
        module.OUTPUT_DIR = build_root / "cluster_performance_outputs"
        module.OUTPUT_FIG = module.OUTPUT_DIR / "fig_bad_basins_clusterwise_nse_lt_0p5_map.png"
        if hasattr(module, "CLUSTER_COLORS"):
            module.CLUSTER_COLORS = dict(CLUSTER_COLORS)
    return _patch


def patch_list_bad_module(build_root: Path):
    def _patch(module):
        orig_ensure = module.ensure_dir
        target = build_root / "Results" / "Tables_cluster"

        def wrapped(path):
            p = Path(path)
            if str(p).replace("\\", "/").endswith("Results/Tables_cluster"):
                target.mkdir(parents=True, exist_ok=True)
                return target
            return orig_ensure(p)
        module.ensure_dir = wrapped
    return _patch


def patch_v08_output(build_root: Path):
    def _patch(module):
        orig_ensure = module.ensure_dir
        target = build_root / "Results" / "Tables_cluster"

        def wrapped(path):
            p = Path(path)
            if str(p).replace("\\", "/").endswith("Results/Tables_cluster"):
                target.mkdir(parents=True, exist_ok=True)
                return target
            return orig_ensure(p)
        module.ensure_dir = wrapped
        if hasattr(module, "CLUSTER_COLORS"):
            module.CLUSTER_COLORS = dict(CLUSTER_COLORS)
    return _patch


def patch_v09_output(build_root: Path, project_root: Path):
    def _patch(module):
        module.OUTPUT_SUBDIR = rel_to_project(build_root / "Results" / "Tables_cluster", project_root)
        if hasattr(module, "CLUSTER_COLORS"):
            module.CLUSTER_COLORS = dict(CLUSTER_COLORS)
    return _patch


def try_run_pro_cluster_map(project_root: Path, build_root: Path, dpi: int) -> bool:
    input_candidates = [
        build_root / "CAMELS_US_hydrology" / "camels_531_master_attributes_clusters.csv",
        project_root / "CAMELS_US_hydrology" / "camels_531_master_attributes_clusters.csv",
        project_root / "camels_531_master_attributes_clusters.csv",
    ]
    input_path = next((p for p in input_candidates if p.exists()), None)
    if input_path is None:
        print("[WARN] Professional cluster map skipped: master attributes/cluster CSV not found yet.")
        return False
    out_png = build_root / "CAMELS_US_hydrology" / "fig01_map_clusters.png"
    try:
        def patch(module):
            if hasattr(module, "CLUSTER_COLORS"):
                module.CLUSTER_COLORS = dict(CLUSTER_COLORS)
        run_module_main(
            "plot_camels_clusters_pro_map",
            ["--input", str(input_path), "--output", str(out_png)],
            patch=patch,
        )
        if not out_png.with_suffix(".pdf").exists() and out_png.exists():
            png_to_pdf(out_png, out_png.with_suffix(".pdf"), dpi=dpi)
        return out_png.exists()
    except Exception as e:
        print(f"[WARN] Professional Cartopy cluster map failed; fallback hydrology map will be used. Reason: {e}")
        return False


def png_to_pdf(png_path: Path, pdf_path: Path, dpi: int = 300) -> None:
    img = plt.imread(png_path)
    h, w = img.shape[:2]
    fig_w, fig_h = w / dpi, h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(img)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


def compose_fig08(build_root: Path, final_outdir: Path, dpi: int) -> None:
    src_dir = build_root / "Results" / "plot_02_water_year_v03" / "NSE"
    panels = [
        ("Cluster-wise", src_dir / "Box_basins_per_year__Cluster-wise__NSE.png"),
        ("Top10", src_dir / "Box_basins_per_year__Top10__NSE.png"),
        ("Benchmark", src_dir / "Box_basins_per_year__Benchmark__NSE.png"),
    ]
    missing = [str(p) for _, p in panels if not p.exists()]
    if missing:
        print("[WARN] Could not compose fig08; missing panels:")
        for m in missing:
            print("   -", m)
        return
    imgs = [plt.imread(p) for _, p in panels]
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 13.8), constrained_layout=False)
    titles = ["Cluster-wise family ensemble", "Top-10 family ensemble", "Reference benchmark"]
    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=4)
    fig.suptitle("Yearly distributions of basin-wise NSE", fontsize=17, fontweight="bold", y=0.992)
    fig.subplots_adjust(top=0.965, bottom=0.01, hspace=0.02)
    out_pdf = final_outdir / "fig08_Box_basins_per_year__NSE.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)




def find_fallback_source(project_root: Path, final_outdir: Path, build_root: Path, rel_src: str) -> Path | None:
    """Find a source image if a legacy script saved it outside the expected build folder."""
    wanted = Path(rel_src).name
    search_roots = [
        build_root,
        final_outdir / "others",
        project_root / "All_plots" / "others",
        project_root / "results" / "All_plots" / "others",
        project_root / "Results",
        project_root / "CAMELS_US_hydrology",
        project_root / "cluster_performance_outputs",
    ]
    seen = set()
    for root in search_roots:
        try:
            root = root.resolve()
        except Exception:
            continue
        if root in seen or not root.exists():
            continue
        seen.add(root)
        hits = list(root.rglob(wanted))
        if hits:
            # Prefer files inside the current final helper folder, otherwise latest modified.
            hits = sorted(hits, key=lambda x: (str(build_root) not in str(x), -x.stat().st_mtime))
            return hits[0]
    return None

def move_pair_to_final(src_png: Path, dst_png: Path, dpi: int) -> bool:
    if not src_png.exists():
        return False
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    src_pdf = src_png.with_suffix(".pdf")
    dst_pdf = dst_png.with_suffix(".pdf")
    if src_pdf.exists():
        if src_pdf.resolve() != dst_pdf.resolve():
            shutil.move(str(src_pdf), str(dst_pdf))
    else:
        png_to_pdf(src_png, dst_pdf, dpi=dpi)
    return True


def collect_final_outputs(project_root: Path, build_root: Path, final_outdir: Path, dpi: int) -> None:
    final_outdir.mkdir(parents=True, exist_ok=True)
    compose_fig08(build_root, final_outdir, dpi=dpi)
    copied, missing = [], []
    for final_name, rel_src in FINAL_SOURCE_MAP.items():
        dst = final_outdir / final_name
        src = build_root / rel_src
        if not src.exists():
            fb = find_fallback_source(project_root=project_root, final_outdir=final_outdir, build_root=build_root, rel_src=rel_src)
            if fb is not None:
                print(f"[FALLBACK] Found {final_name} source at: {fb}")
                src = fb
        ok = move_pair_to_final(src, dst, dpi=dpi)
        if ok:
            copied.append(final_name)
        else:
            missing.append((final_name, src))
    print(f"[OK] Final collection complete: {len(copied)} figures copied/composed in {final_outdir}")
    if missing:
        print("[WARN] Missing mapped outputs:")
        for name, src in missing:
            print(f"   - {name}: {src}")


def enforce_strict_final_folder(final_outdir: Path) -> None:
    allowed = {Path(x).with_suffix(".pdf").name for x in FINAL_SOURCE_MAP.keys()} | {"fig08_Box_basins_per_year__NSE.pdf"}
    removed = []
    for p in final_outdir.glob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".pdf"} and p.name not in allowed:
            removed.append(p.name)
            p.unlink()
        elif p.is_file() and p.suffix.lower() == ".png":
            removed.append(p.name)
            p.unlink()
    if removed:
        print(f"[STRICT] Removed {len(removed)} non-final or non-PDF file(s) from {final_outdir}.")
    print(f"[STRICT] Allowed final figures: {len(allowed)} PDF files.")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    final_outdir = project_root / args.outdir
    build_root = final_outdir / "others"

    os.environ["CAMELS_PROJECT_ROOT"] = str(project_root)
    os.environ["MPLBACKEND"] = "Agg"

    if args.clean_outdir and final_outdir.exists():
        shutil.rmtree(final_outdir)
    final_outdir.mkdir(parents=True, exist_ok=True)

    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)

    apply_publication_style()
    install_dual_savefig_patch(default_dpi=args.dpi)
    patch_camels_utils(build_root)

    print("[STYLE] Javi Del Ser palette retained:", ", ".join(JAVI_PALETTE))
    print("[OUT] Final PDF folder:", final_outdir)
    print("[TMP] Intermediate helper folder inside All_plots:", build_root)

    try:
        print("[1/9] Basin attribute and cluster-summary figures: fig01, fig02, fig03, fig13, fig14, fig15")
        import camels_us_hydrology_summary_plots as hydro
        hydro.OUTPUT_FOLDER_NAME = rel_to_project(build_root / "CAMELS_US_hydrology", project_root)
        hydro.run_all(project_root)

        if not args.skip_pro_cluster_map:
            print("[2/9] Professional cluster map with national/state/province boundaries for fig01")
            try_run_pro_cluster_map(project_root, build_root, dpi=args.dpi)
        else:
            print("[2/9] Professional cluster map skipped by flag.")

        print("[3/9] Whole-hydrograph basin and seed diagnostics: fig04, fig05, fig23, fig24")
        run_module_main("plot_01_hydrograph_overview_v04", ["--metrics", args.metrics, "--models", args.models, "--dpi", str(args.dpi)])

        print("[4/9] Water-year diagnostics: fig06, fig07, fig08, fig25-fig28")
        run_module_main(
            "plot_02_water_year_v03",
            [
                "--metrics", args.metrics,
                "--models", args.models,
                "--dpi", str(args.dpi),
                "--annotate-heatmap",
                "--seed-years", args.seed_years,
            ],
        )

        print("[5/9] Seasonal diagnostics: fig09")
        run_module_main("plot_03_seasonal_v03", ["--metrics", args.metrics, "--models", args.models, "--dpi", str(args.dpi), "--final-only"])

        print("[6/9] Cluster performance summary: fig10")
        run_module_main("cluster_performance_summary_v08", patch=patch_v08_output(build_root))

        print("[7/9] Low-skill basin table and map: fig11")
        try:
            run_module_main("List_Basins_underperformed", patch=patch_list_bad_module(build_root))
        except Exception as e:
            print(f"[WARN] Low-skill basin table generation failed or already exists: {e}")
        run_module_main("plot_bad_basins_us_map", patch=patch_bad_basins_module(project_root, build_root))

        print("[8/9] Journal-style geographic NSE figure: fig12")
        run_module_main("cluster_performance_summary_v09", patch=patch_v09_output(build_root, project_root))

        if args.skip_deep_diagnostics:
            print("[9/9] Deep diagnostic supplementary figures skipped by flag.")
        else:
            print("[9/9] Deep diagnostic supplementary figures: fig16-fig22")
            run_module_main(
                "deep_diagnostic_camels_nse",
                ["--project-root", str(project_root), "--dpi", str(args.dpi), "--output-subdir", rel_to_project_results(build_root / "Results" / "deep_diagnostic_camels_nse", project_root)],
            )

        collect_final_outputs(project_root, build_root, final_outdir, dpi=args.dpi)
        if args.strict_final_folder:
            enforce_strict_final_folder(final_outdir)
        print("[DONE] All available final figures are in:", final_outdir)
    finally:
        if build_root.exists():
            shutil.rmtree(build_root, ignore_errors=True)
            print("[CLEAN] Removed intermediate helper folder:", build_root)


if __name__ == "__main__":
    main()
