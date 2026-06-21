import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
# import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Linux Libertine', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# fontsize
# plt.rcParams.update({'font.size': 14})

def stretched_marker(marker, xscale=1.0, yscale=1.0, rotate=0):
    """Create a marker that is slightly deformed for better visual separation."""
    marker_style = MarkerStyle(marker)
    transform = marker_style.get_transform() + Affine2D().scale(xscale, yscale).rotate_deg(rotate)
    return marker_style.get_path().transformed(transform)

def get_available_filename(filename):
    """Return a filename that does not overwrite existing plots."""
    path = Path(filename)
    if not path.exists():
        return str(path)

    for i in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return str(candidate)

    raise RuntimeError(f"Cannot find available filename for {filename}")

def save_plot(filename):
    """Save current matplotlib figure without overwriting existing files."""
    output_file = get_available_filename(filename)
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved: {output_file}")
    return output_file

MARKER_STYLE_SEQUENCE = [
    {'marker': 'o'},                                                       # circle
    {'marker': 's'},                                                       # square
    {'marker': stretched_marker('^', yscale=1.12), 'linewidth': 0.8},      # triangle up
    {'marker': stretched_marker('v', xscale=1.20, yscale=0.86),
     'size_scale': 1.08, 'linewidth': 0.9},                                # wide triangle down
    {'marker': stretched_marker('<', xscale=0.86, yscale=1.20),
     'size_scale': 1.08, 'linewidth': 0.9},                                # tall triangle left
    {'marker': stretched_marker('>', xscale=1.20, yscale=0.86),
     'size_scale': 1.08, 'linewidth': 0.9},                                # wide triangle right
    {'marker': 'D', 'linewidth': 0.9},                                      # diamond
    {'marker': stretched_marker('P', rotate=45),
     'size_scale': 1.12, 'linewidth': 0.9},                                # rotated filled plus
    {'marker': stretched_marker('*', rotate=18),
     'size_scale': 1.30, 'linewidth': 0.8},                                # rotated star
    {'marker': 'p', 'size_scale': 1.08, 'linewidth': 0.9},                 # pentagon
    {'marker': stretched_marker('h', xscale=1.22, yscale=0.86),
     'size_scale': 1.10, 'linewidth': 1.0},                                # wide hexagon
    {'marker': stretched_marker('8', xscale=0.86, yscale=1.22),
     'size_scale': 1.10, 'linewidth': 1.0},                                # tall octagon
    {'marker': 'X', 'size_scale': 1.18, 'linewidth': 1.0},                 # filled X
    {'marker': (4, 1, 45), 'size_scale': 1.18, 'linewidth': 1.0},          # four-point star
    {'marker': (5, 1, 0), 'size_scale': 1.20, 'linewidth': 1.0},           # five-point star
]

# Override only the markers that are still easy to confuse in the current plots.
MODEL_MARKER_OVERRIDES = {
    'GSN': {'size_scale': 1.18, 'linewidth': 1.2},
    'RQGNN': {'size_scale': 1.18, 'linewidth': 1.2},
    'XGAD': {'size_scale': 1.25, 'linewidth': 1.3},
    'AG-RWNN': {'size_scale': 1.25, 'linewidth': 1.3},
}

def load_and_process_data():
    """加载并处理CSV数据"""
    auc_df = pd.read_csv('auc_table.csv', index_col=0)
    params_df = pd.read_csv('num_params_table.csv', index_col=0)
    
    data_points = []
    
    for model in auc_df.index:
        for dataset in auc_df.columns:
            auc_val = auc_df.loc[model, dataset]
            param_val = params_df.loc[model, dataset]
            
            if pd.notna(auc_val) and pd.notna(param_val) and auc_val != 'None' and param_val != 'None':
                try:
                    auc_val = float(auc_val)
                    param_val = float(param_val)
                    data_points.append({
                        'model': model,
                        'dataset': dataset,
                        'auc': auc_val,
                        'params': param_val,
                        'log_params': np.log10(param_val),
                        'efficiency': auc_val / np.log10(param_val) if param_val > 1 else auc_val
                    })
                except (ValueError, TypeError):
                    continue
    
    return pd.DataFrame(data_points)

def find_pareto_frontier(df):
    """找到Pareto效率前沿"""
    points = df[['log_params', 'auc']].values
    
    pareto_front = []
    for i, point in enumerate(points):
        is_pareto = True
        for j, other_point in enumerate(points):
            if i != j:
                if (other_point[1] >= point[1] and other_point[0] <= point[0] and 
                    (other_point[1] > point[1] or other_point[0] < point[0])):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_front.append(i)
    
    pareto_df = df.iloc[pareto_front].copy()
    pareto_df = pareto_df.sort_values('log_params')
    
    return pareto_df

def perform_clustering(df, n_clusters=4):
    """执行K-means聚类"""
    features = df[['log_params', 'auc']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    return clusters, kmeans, scaler

def get_cluster_display_specs(centers):
    """Order clusters by quadrant semantics: right-top, left-top, left-bottom, right-bottom."""
    if len(centers) != 4:
        raise ValueError("Cluster display ordering expects exactly 4 clusters")

    cluster_ids = list(range(len(centers)))
    auc_order = sorted(cluster_ids, key=lambda idx: centers[idx, 1], reverse=True)
    high_auc = auc_order[:2]
    low_auc = auc_order[2:]

    high_heavy = max(high_auc, key=lambda idx: centers[idx, 0])
    high_lightweight = min(high_auc, key=lambda idx: centers[idx, 0])
    low_lightweight = min(low_auc, key=lambda idx: centers[idx, 0])
    low_heavy = max(low_auc, key=lambda idx: centers[idx, 0])

    return [
        {'cluster_id': high_heavy, 'label': 'High-Performance Heavy', 'color': 'red'},
        {'cluster_id': high_lightweight, 'label': 'High-Performance Lightweight', 'color': 'blue'},
        {'cluster_id': low_lightweight, 'label': 'Low-Performance Lightweight', 'color': 'green'},
        {'cluster_id': low_heavy, 'label': 'Low-Performance Heavy', 'color': 'purple'},
    ]

def get_model_styles(models):
    """为每个模型分配颜色和图形样式"""
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    colors[-1] = (0, 0, 0, 1)
    colors[-2] = (0.5, 0.5, 0.5, 1)
    # different color schemes
    # colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
    marker_styles = MARKER_STYLE_SEQUENCE
    
    # 确保有足够的标记样式
    if len(models) > len(marker_styles):
        marker_styles = marker_styles * ((len(models) // len(marker_styles)) + 1)
    
    model_styles = {}
    for i, model in enumerate(models):
        style = {
            'color': colors[i],
            'edgecolor': 'black',
            **marker_styles[i],
        }
        style.update(MODEL_MARKER_OVERRIDES.get(model, {}))
        model_styles[model] = style
    
    return model_styles

def scatter_model_points(ax, x, y, style, base_size, base_linewidth, **kwargs):
    """Scatter points with per-model marker size and outline overrides."""
    return ax.scatter(
        x,
        y,
        c=[style['color']],
        marker=style['marker'],
        s=base_size * style.get('size_scale', 1.0),
        edgecolors=style.get('edgecolor', 'black'),
        linewidths=style.get('linewidth', base_linewidth),
        **kwargs
    )

def create_model_legend_handles(models, model_styles, marker_size=7):
    """Create compact legend handles for model marker styles."""
    handles = []
    for model in models:
        style = model_styles[model]
        handles.append(Line2D(
            [0], [0],
            marker=style['marker'],
            linestyle='None',
            label=model,
            markerfacecolor=style['color'],
            markeredgecolor=style.get('edgecolor', 'black'),
            markeredgewidth=style.get('linewidth', 0.8),
            markersize=marker_size * np.sqrt(style.get('size_scale', 1.0)),
            alpha=0.85,
        ))
    return handles

def create_basic_scatter_plot(df):
    """创建基本散点图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = df['model'].unique()
    model_styles = get_model_styles(models)
    
    for model in models:
        model_data = df[df['model'] == model]
        scatter_model_points(
            ax, model_data['log_params'], model_data['auc'], model_styles[model],
            base_size=80, base_linewidth=0.5, label=model, alpha=0.7
        )
    
    ax.set_xlabel('Log10(Parameters/k)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Model Parameters vs AUC Performance', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot('scatter_plot.pdf')
    plt.close()

def create_pareto_frontier_plot(df):
    """创建Pareto前沿分析图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = df['model'].unique()
    model_styles = get_model_styles(models)
    
    for model in models:
        model_data = df[df['model'] == model]
        scatter_model_points(
            ax, model_data['log_params'], model_data['auc'], model_styles[model],
            base_size=50, base_linewidth=0.3, alpha=0.6, label=model
        )
    
    pareto_df = find_pareto_frontier(df)
    ax.plot(pareto_df['log_params'], pareto_df['auc'], 
             'r-', linewidth=3, label='Pareto Frontier', zorder=10)
    ax.scatter(pareto_df['log_params'], pareto_df['auc'], 
               c='red', marker='o', s=100, edgecolors='darkred', 
               linewidth=2, zorder=11, alpha=0.8)
    
    for idx, row in pareto_df.iterrows():
        ax.annotate(f"{row['model']}", 
                    (row['log_params'], row['auc']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=10, alpha=0.9, fontweight='bold')
    
    ax.set_xlabel('Log10(Parameters/k)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Pareto Efficiency Frontier Analysis', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot('pareto_frontier.pdf')
    plt.close()

def create_clustering_plot(df, rotate=False):
    """创建K-means聚类分析图"""
    if rotate:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    clusters, kmeans, scaler = perform_clustering(df, n_clusters=4)
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    models = df['model'].unique()
    model_styles = get_model_styles(models)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_specs = get_cluster_display_specs(centers)
    
    # 左图：按聚类显示，用颜色区分聚类
    for order, spec in enumerate(cluster_specs, start=1):
        cluster_id = spec['cluster_id']
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        ax1.scatter(cluster_data['log_params'], cluster_data['auc'],
                   c=spec['color'], label=f"{order}. {spec['label']}",
                   alpha=0.7, s=120, edgecolors='black', linewidth=0.5)
    
    ax1.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', 
               s=200, linewidths=4, label='Cluster Centers')
    
    ax1.set_xlabel('Log10(Parameters/k)', fontsize=16)
    ax1.set_ylabel('AUC', fontsize=16)
    ax1.set_title('Clustering by Groups', fontsize=16)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：按模型显示，用图形+颜色区分模型，背景色显示聚类
    for spec in cluster_specs:
        cluster_id = spec['cluster_id']
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        # 添加背景色区域（可选）
        if len(cluster_data) > 2:
            hull_points = cluster_data[['log_params', 'auc']].values
            if len(hull_points) >= 3:
                try:
                    hull = ConvexHull(hull_points)
                    for simplex in hull.simplices:
                        ax2.fill(hull_points[simplex, 0], hull_points[simplex, 1], 
                                alpha=0.7, color=spec['color'], linewidth=2)
                except:
                    pass
    
    for model in models:
        model_data = df_clustered[df_clustered['model'] == model]
        scatter_model_points(
            ax2, model_data['log_params'], model_data['auc'], model_styles[model],
            base_size=120, base_linewidth=1, label=model, alpha=0.8
        )
    
    ax2.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', 
               s=200, linewidths=4, label='Centers')
    
    ax2.set_xlabel('Log10(Parameters/k)', fontsize=16)
    # ax2.set_ylabel('AUC', fontsize=16)
    ax2.set_title('Models within Clusters', fontsize=16)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'clustering_analysis_ro.pdf' if rotate else 'clustering_analysis.pdf'
    save_plot(filename)
    plt.close()
    
    return df_clustered

def create_efficiency_plot(df):
    """创建参数效率分析图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = df['model'].unique()
    model_styles = get_model_styles(models)
    
    efficiency_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    x_range = np.linspace(df['log_params'].min(), df['log_params'].max(), 100)
    
    for eff in efficiency_levels:
        y_eff = eff * x_range
        ax.plot(x_range, y_eff, '--', alpha=0.8, 
                label=f'Efficiency={eff:.2f}', color='gray')
    
    for model in models:
        model_data = df[df['model'] == model]
        scatter_model_points(
            ax, model_data['log_params'], model_data['auc'], model_styles[model],
            base_size=80, base_linewidth=0.5, alpha=0.7, label=model
        )
    
    ax.set_xlabel('Log10(Parameters/k)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Parameter Efficiency Analysis (AUC/Log10(Params))', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    # limit y-axis to [0.2, 1.2]
    ax.set_ylim(0.2, 1.2)
    
    plt.tight_layout()
    save_plot('efficiency_analysis.pdf')
    plt.close()

def create_dataset_faceted_pareto_plot(df):
    """按数据集分别展示参数量-性能散点，并为每个数据集画Pareto前沿。"""
    datasets = df['dataset'].unique()
    models = df['model'].unique()
    model_styles = get_model_styles(models)
    n_cols = 4
    n_rows = int(np.ceil(len(datasets) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.3 * n_cols, 3.25 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).flatten()

    x_margin = 0.08 * (df['log_params'].max() - df['log_params'].min())
    y_margin = 0.05 * (df['auc'].max() - df['auc'].min())
    x_lim = (df['log_params'].min() - x_margin, df['log_params'].max() + x_margin)
    y_lim = (max(0, df['auc'].min() - y_margin), min(1.05, df['auc'].max() + y_margin))

    for ax, dataset in zip(axes, datasets):
        dataset_data = df[df['dataset'] == dataset]
        for model in models:
            model_data = dataset_data[dataset_data['model'] == model]
            if model_data.empty:
                continue
            scatter_model_points(
                ax,
                model_data['log_params'],
                model_data['auc'],
                model_styles[model],
                base_size=42,
                base_linewidth=0.35,
                alpha=0.78,
                zorder=3,
            )

        pareto_df = find_pareto_frontier(dataset_data)
        if len(pareto_df) > 1:
            ax.plot(
                pareto_df['log_params'],
                pareto_df['auc'],
                color='crimson',
                linewidth=1.8,
                alpha=0.85,
                zorder=4,
            )
        ax.scatter(
            pareto_df['log_params'],
            pareto_df['auc'],
            facecolors='none',
            edgecolors='crimson',
            marker='o',
            s=80,
            linewidths=1.3,
            zorder=5,
        )
        for _, row in pareto_df.iterrows():
            ax.annotate(
                row['model'],
                (row['log_params'], row['auc']),
                xytext=(3, 4),
                textcoords='offset points',
                fontsize=7,
                fontweight='bold' if row['model'] == 'AG-RWNN' else 'normal',
                alpha=0.92,
            )

        ax.set_title(dataset, fontsize=11)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(datasets):]:
        ax.axis('off')

    for ax in axes[-n_cols:]:
        ax.set_xlabel('Log10(Parameters/k)', fontsize=10)
    for ax in axes[::n_cols]:
        ax.set_ylabel('AUC', fontsize=10)

    handles = create_model_legend_handles(models, model_styles, marker_size=6)
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.035),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle('Dataset-wise Parameter-Performance Pareto Frontiers', fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    save_plot('dataset_faceted_pareto.pdf')
    plt.close()

def add_dataset_relative_metrics(df):
    """Add per-dataset normalized AUC, ranks, and Pareto membership."""
    metric_frames = []
    for _, dataset_data in df.groupby('dataset', sort=False):
        group = dataset_data.copy()
        best_auc = group['auc'].max()
        worst_auc = group['auc'].min()
        auc_range = best_auc - worst_auc
        group['auc_norm'] = (group['auc'] - worst_auc) / auc_range if auc_range > 0 else 1.0
        group['auc_delta_best'] = best_auc - group['auc']
        group['auc_rank'] = group['auc'].rank(ascending=False, method='min')
        group['is_pareto'] = False
        pareto_idx = find_pareto_frontier(group).index
        group.loc[pareto_idx, 'is_pareto'] = True
        metric_frames.append(group)

    return pd.concat(metric_frames, ignore_index=False)

def create_model_tradeoff_summary_plot(df):
    """汇总模型级参数量、归一化性能和Pareto出现次数。"""
    df_metrics = add_dataset_relative_metrics(df)
    summary = df_metrics.groupby('model', sort=False).agg(
        median_params=('params', 'median'),
        median_log_params=('log_params', 'median'),
        mean_auc_norm=('auc_norm', 'mean'),
        std_auc_norm=('auc_norm', 'std'),
        mean_auc_rank=('auc_rank', 'mean'),
        pareto_count=('is_pareto', 'sum'),
        n_datasets=('dataset', 'nunique'),
    ).reset_index()
    summary['std_auc_norm'] = summary['std_auc_norm'].fillna(0)

    models = df['model'].unique()
    model_styles = get_model_styles(models)
    fig, (ax_scatter, ax_bar) = plt.subplots(
        1, 2,
        figsize=(15, 6),
        gridspec_kw={'width_ratios': [1.5, 1]},
    )

    for _, row in summary.iterrows():
        style = model_styles[row['model']]
        ax_scatter.errorbar(
            row['median_log_params'],
            row['mean_auc_norm'],
            yerr=row['std_auc_norm'],
            color=style['color'],
            alpha=0.28,
            capsize=2,
            linewidth=0.8,
            zorder=2,
        )
        scatter_model_points(
            ax_scatter,
            [row['median_log_params']],
            [row['mean_auc_norm']],
            style,
            base_size=80 + 36 * row['pareto_count'],
            base_linewidth=0.7,
            alpha=0.82,
            zorder=3,
        )
        ax_scatter.annotate(
            row['model'],
            (row['median_log_params'], row['mean_auc_norm']),
            xytext=(5, 4),
            textcoords='offset points',
            fontsize=8,
            fontweight='bold' if row['model'] == 'AG-RWNN' else 'normal',
            alpha=0.92,
        )

    ax_scatter.set_xlabel('Median Log10(Parameters/k)', fontsize=11)
    ax_scatter.set_ylabel('Mean normalized AUC within dataset', fontsize=11)
    ax_scatter.set_title('Cross-dataset parameter-performance tradeoff', fontsize=12)
    ax_scatter.set_ylim(-0.04, 1.08)
    ax_scatter.grid(True, alpha=0.28)

    summary_sorted = summary.sort_values(
        ['pareto_count', 'mean_auc_norm', 'mean_auc_rank'],
        ascending=[False, False, True],
    )
    y_pos = np.arange(len(summary_sorted))
    bar_colors = [model_styles[model]['color'] for model in summary_sorted['model']]
    ax_bar.barh(y_pos, summary_sorted['pareto_count'], color=bar_colors, alpha=0.82, edgecolor='black', linewidth=0.4)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(summary_sorted['model'], fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Pareto-front appearances', fontsize=11)
    ax_bar.set_title('How often each model is Pareto-efficient', fontsize=12)
    ax_bar.set_xlim(0, max(1, int(summary_sorted['pareto_count'].max())) + 1)
    ax_bar.grid(True, axis='x', alpha=0.25)

    for y, (_, row) in enumerate(summary_sorted.iterrows()):
        ax_bar.text(
            row['pareto_count'] + 0.08,
            y,
            f"rank={row['mean_auc_rank']:.1f}",
            va='center',
            fontsize=7,
            alpha=0.75,
        )

    plt.tight_layout()
    save_plot('model_tradeoff_summary.pdf')
    plt.close()

def create_visualization(df):
    """创建所有可视化图表"""
    print("Creating visualizations...")
    create_basic_scatter_plot(df)
    create_pareto_frontier_plot(df)
    df_clustered = create_clustering_plot(df)
    create_efficiency_plot(df)
    create_dataset_faceted_pareto_plot(df)
    create_model_tradeoff_summary_plot(df)
    return df_clustered

def print_analysis_summary(df, df_clustered):
    """打印分析摘要"""
    print("=== 数据分析摘要 ===")
    print(f"总数据点数: {len(df)}")
    print(f"模型数量: {df['model'].nunique()}")
    print(f"数据集数量: {df['dataset'].nunique()}")
    print(f"参数范围: {df['params'].min():.1f}k - {df['params'].max():.1f}k")
    print(f"AUC范围: {df['auc'].min():.3f} - {df['auc'].max():.3f}")
    
    print("\n=== Pareto前沿模型 ===")
    pareto_df = find_pareto_frontier(df)
    for idx, row in pareto_df.iterrows():
        print(f"{row['model']}: {row['params']:.1f}k参数, AUC={row['auc']:.3f}")
    
    print("\n=== 聚类分析结果 ===")
    for i in range(4):
        cluster_data = df_clustered[df_clustered['cluster'] == i]
        avg_params = cluster_data['params'].mean()
        avg_auc = cluster_data['auc'].mean()
        models_in_cluster = cluster_data['model'].unique()
        print(f"群组 {i+1}: 平均{avg_params:.1f}k参数, 平均AUC={avg_auc:.3f}")
        print(f"  包含模型: {', '.join(models_in_cluster)}")
    
    print("\n=== 高效率模型 (Top 5) ===")
    top_efficient = df.nlargest(5, 'efficiency')
    for idx, row in top_efficient.iterrows():
        print(f"{row['model']} ({row['dataset']}): 效率={row['efficiency']:.3f}")

def main():
    """主函数"""
    df = load_and_process_data()
    print(f"Loaded {len(df)} data points from CSV files")
    
    df_clustered = create_visualization(df)
    print_analysis_summary(df, df_clustered)
    
    print("\nAll PDF files have been saved successfully!")
    print("Generated files (auto-suffixed when the base filename already exists):")
    print("- scatter_plot.pdf: Basic scatter plot with model colors")
    print("- pareto_frontier.pdf: Pareto efficiency frontier analysis")
    print("- clustering_analysis.pdf: K-means clustering (k=4)")
    print("- efficiency_analysis.pdf: Parameter efficiency analysis")
    print("- dataset_faceted_pareto.pdf: Dataset-wise Pareto frontiers")
    print("- model_tradeoff_summary.pdf: Cross-dataset normalized tradeoff summary")

if __name__ == "__main__":
    main()
