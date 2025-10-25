import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
import os

# 定义绘图函数
def plot_correlation(x, y, save_directory, filename, xlabel='Measured', ylabel='Predicted'):
    # Kernel density estimate
    kernel = gaussian_kde(np.vstack([x, y]))
    c = kernel(np.vstack([x, y]))  # 计算密度

    # Figure configurations 
    sns.set_theme(style='ticks', font_scale=0.75, rc={
        'svg.fonttype': 'none',
        'font.sans-serif': ['Arial'],
        'font.family': 'sans-serif',
        'text.usetex': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'lines.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'legend.fontsize': 9,
        'legend.title_fontsize': 9,
        'xtick.major.size': 3,
        'xtick.major.pad': 3,
        'xtick.major.width': 0.5,
        'ytick.major.size': 3,
        'ytick.major.pad': 3,
        'ytick.major.width': 0.5,
    })

    pt_size = 0.2
    cmap = mpl.cm.inferno

    # Axis limits
    xlim = [-13.5, 9.5]
    xticks = [-10, -5, 0, 5]

    fig = plt.figure(figsize=(1.3, 1.3), dpi=300)
    gs = fig.add_gridspec(
        1, 1, left=0.26, right=0.9, bottom=0.21, top=0.88
    )
    gs2 = fig.add_gridspec(
        1, 1, left=0.91, right=0.94, bottom=0.21, top=0.88
    )

    # BOTH
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(
        x, y, c=c, s=pt_size, cmap=cmap, 
        rasterized=True, edgecolors=None, linewidth=0
    )
    ax.set_xlim(xlim); ax.set_ylim(xlim)
    ax.set_xticks(xticks); ax.set_yticks(xticks)
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=0)

    # Correlation text
    ax.text(
        x=0.05, y=0.95, s=r'$\rho$ = {:.3f}'.format(np.corrcoef(x, y)[0, 1]), 
        transform=ax.transAxes, ha='left', va='top', fontsize=7
    )

    ax.tick_params(axis='x', labelsize=7, length=2, pad=2)
    ax.tick_params(axis='y', labelsize=7, length=2, pad=1)

    # Colorbar
    ax = fig.add_subplot(gs2[0, 0])
    norm = mpl.colors.Normalize(vmin=np.min(c), vmax=np.max(c))
    cb = mpl.colorbar.ColorbarBase(
        ax, 
        cmap=cmap,
        norm=norm,
        orientation='vertical'
    )
    cb.set_ticks([])
    # ax.text(1, 1.02, 'Density', transform=ax.transAxes, ha='right', va='bottom')

    # 保存图像
    fig_filename = os.path.join(save_directory, filename)
    fig.savefig(f'{fig_filename}.png')
    fig.savefig(f'{fig_filename}_600dpi.svg', dpi=600)
    fig.savefig(f'{fig_filename}_1200dpi.svg', dpi=1200)

    plt.close()

    print(f'图像已保存至: {fig_filename}.png')
