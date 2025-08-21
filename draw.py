import numpy as np
import matplotlib.pyplot as plt

def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None,
         grid=True):
    """
    绘制数据曲线
    
    参数:
        X: x轴数据或y轴数据(当Y为None时)
        Y: y轴数据，可为None或多维数组(多条曲线)
        xlabel: x轴标签
        ylabel: y轴标签
        legend: 图例列表
        xlim: x轴范围，如(xmin, xmax)
        ylim: y轴范围，如(ymin, ymax)
        xscale: x轴比例尺，'linear'或'log'
        yscale: y轴比例尺，'linear'或'log'
        fmts: 线条样式列表，循环使用
        figsize: 图形大小，(宽度, 高度)
        axes: 可选的matplotlib轴对象，用于子图绘制
        
    返回:
        axes: matplotlib轴对象
    """
    # 处理输入数据
    if Y is None:
        Y = X
        X = np.arange(len(Y))
    # 转换为numpy数组便于处理
    X = np.asarray(X)
    Y = np.asarray(Y)
    # 如果Y是1D数组，转换为2D以便统一处理
    if Y.ndim == 1: Y = Y.reshape(-1, 1)
    # 创建或获取轴对象
    if axes is None: _, axes = plt.subplots(figsize=figsize)
    # 绘制每条曲线
    n = Y.shape[1]  # 曲线数量
    for i in range(n):
        if X.ndim == 1:
            x = X
        else:
            x = X[:, i]  # 每条曲线有自己的x数据
        axes.plot(x, Y[:, i], fmts[i % len(fmts)])
    
    # 设置坐标轴标签
    if xlabel: axes.set_xlabel(xlabel)
    if ylabel: axes.set_ylabel(ylabel)
    # 设置坐标轴范围
    if xlim: axes.set_xlim(xlim)
    if ylim: axes.set_ylim(ylim)
    # 设置坐标轴比例尺
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    # 设置图例
    if legend: axes.legend(legend)
    if grid: axes.grid(True, linestyle='--', alpha=0.7)  # 虚线网格，半透明
    # 调整布局
    plt.tight_layout()
    
    return axes

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        
        
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib"""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        if legend:
            ax.legend(legend)
        ax.grid()

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)
        self.config_axes()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# x = np.linspace(0, 10, 1000)  # 增加数据点数量并扩大范围
# y1 = np.sin(x)
# y2 = np.cos(x)
# y3 = np.sin(x) * np.cos(x)

# # 绘制单条曲线 - 使用适合数据的xlim
# plot(x, y1, xlabel='x', ylabel='sin(x)', legend=['sin(x)'], 
#      xlim=[0, 10], ylim=[-1, 1])  # 这个xlim现在与数据范围匹配
# plt.show()

# 同时绘制多条曲线
# plot(x, [y1, y2, y3], xlabel='x', ylabel='Values', 
#      legend=['sin(x)', 'cos(x)', 'sin(x)cos(x)'],
#      xlim=[0, 10], ylim=[-1, 1], figsize=(8, 5))
# plt.show()
# # 绘制多条曲线
# plot(x, [y1, y2, y3], xlabel='x', ylabel='value', 
#         legend=['sin(x)', 'cos(x)', 'sin(x)cos(x)'],
#         figsize=(8, 5))
# plt.show()
    
# # 绘制无X数据的曲线（使用索引作为X）
# plot(y1, xlabel='index', ylabel='sin(x)', legend=['sin(x)'])
# plt.show()
    
# # 对数坐标示例
# x_log = np.linspace(1, 10, 100)
# y_log = np.exp(x_log)
# plot(x_log, y_log, xlabel='x', ylabel='exp(x)', 
#         xscale='log', yscale='log', legend=['exp(x)'])
# plt.show()
