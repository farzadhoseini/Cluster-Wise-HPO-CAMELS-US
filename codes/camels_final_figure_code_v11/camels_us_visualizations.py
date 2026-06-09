# camels_us_visualizations.py
# ============================
"""
CAMELS-US Visualization Module

Requirements:
    pip install numpy pandas matplotlib seaborn scipy geopandas shapely
    pip install plotly panel joypy skill_metrics hvplot holoviews panel

Optional for Dash (if needed):
    pip install dash==2.9.3 werkzeug==2.2.3

Usage:
    Import this module and call any function with your dataframes or arrays,
    or run `run_all_visualizations` to generate and save all plots automatically.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde
import geopandas as gpd

# Default directory for saving plots (overrideable)
default_save_dir = r"F:\Experiments\CAMELS_US\TrainsTests\Results\Plots"

# Try Dash import; disable if incompatible
dash_available = True
try:
    import dash
    from dash import dcc, html, Input, Output
    import plotly.express as px
except Exception:
    dash_available = False
    def launch_dashboard(*args, **kwargs):
        print("Dash is not available. To enable, install compatible versions: pip install dash==2.9.3 werkzeug==2.2.3")

# === 1. Boxplots and Violin Plots ===
def plot_box_violin(metrics_df, metric, by='model', kind='violin', save_path=None):
    plt.figure(figsize=(8, 6))
    if kind == 'violin':
        sns.violinplot(x=by, y=metric, data=metrics_df, inner='quartile')
    else:
        sns.boxplot(x=by, y=metric, data=metrics_df)
    sns.stripplot(x=by, y=metric, data=metrics_df, color='k', size=2, alpha=0.4)
    plt.xlabel(by)
    plt.ylabel(metric)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# === 2. Taylor Diagram ===
def plot_taylor_diagram(obs, sims_dict, save_path=None):
    from skill_metrics import taylor_diagram
    fig = plt.figure(figsize=(8, 6))
    taylor_diagram(obs, list(sims_dict.values()), 1, fig=fig,
                   markerLabel=list(sims_dict.keys()))
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# === 3. Flow Duration Curve (FDC) ===
def plot_fdc(obs_flow, sim_flow=None, label_obs='Observed', label_sim='Modeled', save_path=None):
    def get_fdc(arr):
        sorted_arr = np.sort(arr)
        exceed = np.arange(1., len(arr)+1)[::-1] / len(arr)
        return sorted_arr, exceed
    obs_x, obs_y = get_fdc(obs_flow)
    plt.figure(figsize=(8, 6))
    plt.plot(obs_y, obs_x, label=label_obs)
    if sim_flow is not None:
        sim_x, sim_y = get_fdc(sim_flow)
        plt.plot(sim_y, sim_x, label=label_sim)
    plt.yscale('log')
    plt.xlabel('Exceedance Probability')
    plt.ylabel('Flow')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# === 4. Spatial Map of Performance ===
def plot_spatial_performance(basin_info_df, metric, save_path=None):
    gdf = gpd.GeoDataFrame(
        basin_info_df,
        geometry=gpd.points_from_xy(basin_info_df.lon, basin_info_df.lat),
        crs='EPSG:4326'
    )
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fig, ax = plt.subplots(figsize=(10, 6))
    world.plot(ax=ax, color='lightgray')
    gdf.plot(column=metric, ax=ax, legend=True, markersize=20, cmap='viridis')
    plt.title(f'Spatial Distribution of {metric}')
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# === 5. Radar (Spider) Chart ===
def plot_radar(models_metrics, metrics, save_path=None):
    labels = metrics
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))
    for model, vals in models_metrics.items():
        data = vals + vals[:1]
        ax.plot(angles, data, label=model)
        ax.fill(angles, data, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, max(max(v) for v in models_metrics.values()))
    ax.legend(loc='upper right')
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# === 6. Seasonal / Monthly Boxplots ===
def plot_seasonal_boxplot(ts_df, value_col='flow', time_col='date', model_col='model', save_path=None):
    df = ts_df.copy()
    df['month'] = pd.to_datetime(df[time_col]).dt.month
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='month', y=value_col, hue=model_col, data=df)
    plt.xlabel('Month')
    plt.ylabel(value_col)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# === 7. Scatter-Matrix (Pairplot) ===
def plot_pairplot(metrics_df, vars_list, hue='model', save_path=None):
    sns.pairplot(metrics_df, vars=vars_list, hue=hue, diag_kind='kde')
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# === 8. Error vs. Basin Attributes ===
def plot_error_vs_attribute(metrics_df, attr_df, metric, attribute, save_path=None):
    df = metrics_df[[metric]].join(attr_df[[attribute]] )
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=attribute, y=metric, data=df)
    sns.regplot(x=attribute, y=metric, data=df, scatter=False, color='r')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# === 9. Ridgeline (Joy) Plots ===
def plot_ridgeline(metrics_df, by, metric, save_path=None):
    import joypy
    fig, axes = joypy.joyplot(metrics_df, by=by, column=metric, figsize=(8, 6))
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# === 10. Interactive Dashboard using Panel ===
def launch_dashboard_panel(metrics_df, basin_info_df):
    import panel as pn
    import holoviews as hv
    import hvplot.pandas  # noqa
    pn.extension()
    merged = metrics_df.merge(basin_info_df, on='basin')
    scatter = merged.hvplot.points('lon', 'lat', geo=True, size=5, hover_cols=['basin','NSE'], color='NSE', cmap='viridis')
    hist = merged.hvplot.hist('NSE', bins=30)
    pn.Column(pn.pane.HoloViews(scatter), pn.pane.HoloViews(hist)).servable()

# === 11. Heatmap of Model Differences ===
def plot_heatmap_diff(metrics_df, baseline='Benchmark', save_path=None):
    diff = metrics_df.subtract(metrics_df[baseline], axis=0)
    plt.figure(figsize=(10, 12))
    sns.heatmap(diff, cmap='bwr', center=0)
    plt.title(f'Difference from {baseline}')
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# === 12. Time-Series Skill Diagrams ===
def plot_timeseries_skill(obs_ts, sim_ts_dict, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(obs_ts.index, obs_ts.values, label='Observed', color='k')
    for label, ts in sim_ts_dict.items():
        plt.plot(ts.index, ts.values, alpha=0.7, label=label)
    plt.ylabel('Flow')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# === Automated Runner for all visualizations ===
def run_all_visualizations(test_metrics_dicts,
                           basin_info_df=None,
                           ts_data_dict=None,
                           save_dir=None):
    """
    Generate and save all 12 visualizations for CAMELS-US data.

    Parameters:
    - test_metrics_dicts: dict of model->{basin->{site->{metric: value}}}
    - basin_info_df: DataFrame with ['basin','lat','lon'] (optional)
    - ts_data_dict: dict of model->{basin: DataFrame with ['date','flow']} (optional)
    - save_dir: directory to write plots
    """
    # Determine save directory
    out_dir = save_dir or default_save_dir
    os.makedirs(out_dir, exist_ok=True)

    # Flatten metrics into DataFrame
    rows = []
    for model, basins in test_metrics_dicts.items():
        for basin, sites in basins.items():
            for site, mets in sites.items():
                r = {'model': model, 'basin': basin, 'site': site}
                r.update(mets)
                rows.append(r)
    metrics_df = pd.DataFrame(rows)
    metrics = [c for c in metrics_df.columns if c not in ['model','basin','site']]

    # 1. PDFs from previous code (pass out_dir)
    for metric in metrics:
        fname = f'CAMELS_US_PDF_{metric}_v01.jpg'
        create_single_metric_pdf(
            test_metrics_dicts, metric,
            use_kde=False, bins=60,
            save_dir=out_dir
        )

    # 2 & 3. Box and Violin
    for metric in metrics:
        path_box = os.path.join(out_dir, f'CAMELS_US_Boxplot_{metric}.jpg')
        plot_box_violin(metrics_df, metric, by='model', kind='box', save_path=path_box)
        path_violin = os.path.join(out_dir, f'CAMELS_US_Violin_{metric}.jpg')
        plot_box_violin(metrics_df, metric, by='model', kind='violin', save_path=path_violin)

    # 4. Radar
    models_metrics = {m: metrics_df[metrics][metrics_df.model==m].mean().tolist() for m in metrics_df.model.unique()}
    path_radar = os.path.join(out_dir, 'CAMELS_US_Radar.jpg')
    plot_radar(models_metrics, metrics, save_path=path_radar)

    # 5. Pairplot
    path_pair = os.path.join(out_dir, 'CAMELS_US_Pairplot.jpg')
    plot_pairplot(metrics_df, metrics, save_path=path_pair)

    # 6. Ridgeline
    for metric in metrics:
        path_ridge = os.path.join(out_dir, f'CAMELS_US_Ridgeline_{metric}.jpg')
        plot_ridgeline(metrics_df, by='model', metric=metric, save_path=path_ridge)

    # 7. Heatmap diff (first metric example)
    wide = metrics_df.groupby(['basin','model'])[metrics[0]].mean().unstack()
    path_heat = os.path.join(out_dir, f'CAMELS_US_Heatmap_{metrics[0]}_Diff.jpg')
    plot_heatmap_diff(wide, baseline='Benchmark', save_path=path_heat)

    # 8. Spatial (requires basin_info_df)
    if basin_info_df is not None:
        for metric in metrics:
            dfm = basin_info_df.copy()
            dfm[metric] = metrics_df.groupby('basin')[metric].mean().reindex(dfm.basin).values
            path_sp = os.path.join(out_dir, f'CAMELS_US_Spatial_{metric}.jpg')
            plot_spatial_performance(dfm, metric, save_path=path_sp)

    # 9. Seasonal / Monthly (requires ts_data_dict)
    if ts_data_dict is not None:
        ts_rows=[]
        for model, basins in ts_data_dict.items():
            for basin, df in basins.items():
                dfc = df.copy()
                dfc['model'] = model; dfc['basin'] = basin
                ts_rows.append(dfc)
        ts_df = pd.concat(ts_rows, ignore_index=True)
        path_season = os.path.join(out_dir, 'CAMELS_US_Seasonal_flow.jpg')
        plot_seasonal_boxplot(ts_df, value_col='flow', time_col='date', model_col='model', save_path=path_season)

    # 10 & 11. Time-series and FDC (requires ts_data_dict)
    if ts_data_dict is not None:
        for basin in ts_data_dict.get('RO', {}):
            obs = ts_data_dict['RO'][basin].set_index('date')['flow']
            sims = {m: ts_data_dict[m][basin].set_index('date')['flow'] for m in test_metrics_dicts if basin in ts_data_dict.get(m, {})}
            path_ts = os.path.join(out_dir, f'CAMELS_US_Timeseries_{basin}.jpg')
            plot_timeseries_skill(obs, sims, save_path=path_ts)
            for m, sim_ts in sims.items():
                arr_obs = obs.values
                arr_sim = sim_ts.values
                path_fdc = os.path.join(out_dir, f'CAMELS_US_FDC_{basin}_{m}.jpg')
                plot_fdc(arr_obs, arr_sim, label_obs='Obs', label_sim=m, save_path=path_fdc)

if __name__ == '__main__':
    pass