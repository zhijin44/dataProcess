import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Affine2D

text_size = 18
ticks_size = 14

def heatmap_to_coverage(file, threshold):
    df = pd.read_csv(os.path.join("./", file), delimiter=' ', skiprows=3, header=None, usecols=[5])
    df['binary'] = np.where(df[5] >= threshold, 1, 0)
    matrix = df['binary'].values.reshape(20, 20)
    print(matrix)

    # 创建自定义颜色映射
    colors = ["lightgreen", "lightyellow"]
    cmap = mcolors.ListedColormap(colors)

    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap=cmap, interpolation='nearest')

    # 添加更明显的黑色边框
    rect = Rectangle((-0.5, -0.5), matrix.shape[1], matrix.shape[0], linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # 关闭轴
    ax.axis('off')

    # 保存图形，并提高DPI来保持清晰度
    # if not os.path.exists("./img"):
    #     os.makedirs("./img")
    # plt.savefig(os.path.join("./img", "heatmap.png"), bbox_inches='tight', pad_inches=0, dpi=300)

    # 显示图形
    # 显示图形
    plt.show()

# heatmap_to_coverage("1.000e-03_5.00_wifi.p2m", -94)
# heatmap_to_coverage("1.000e-03_5.00_5g.p2m", -78)

def create_setting():
    # 创建一个 20 x 20 的全零矩阵
    matrix = np.zeros((20, 20), dtype=int)

    # 修改矩阵的特定区域的值
    matrix[5:10, 0:8] = 1
    matrix[2:13, 13:18] = 1
    # matrix = np.fliplr(matrix)
    # matrix[5:10, 0:8] = 1

    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8))

    # trans = Affine2D().translate(-10, -10).rotate_deg(50).scale(1, 0.5).translate(10, 10)
    trans = (Affine2D()
             .translate(-10, -10)  # 将中心移至原点
             .rotate_deg(130)
             .scale(1, 0.5)
             .translate(10, 10))  # 将中心移回原位置
    # 应用仿射变换
    ax.matshow(matrix, cmap=plt.cm.colors.ListedColormap(['lightgray', '#0c84c6']), transform=trans + ax.transData)

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(-5, 25)  # 调整视图范围，确保整个图像可见
    ax.set_ylim(-5, 30)  # 调整视图范围，确保整个图像可见

    # 保存图表
    plt.savefig('skewed_output_5g_50.png', transparent=True)

    # 展示图表
    plt.show()


# create_setting()


def draw_accuracy():
    # 数据
    rate = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    accuracy_str = ['77.61%', '84.31%', '87.76%', '90.52%', '92.09%', '93.17%', '94.75%']
    accuracy = [float(acc.strip('%')) for acc in accuracy_str]
    errors = [0.5, 0.35, 0.3, 0.28, 0.4, 0.25, 0.35]

    # 定义x轴的位置
    x_positions = np.arange(len(rate))

    # 绘制柱状图
    bar_color = '#ffa510'
    bar_width = 0.4  # 调整此值以修改柱宽度
    plt.figure(figsize=(10, 5))
    plt.bar(x_positions, accuracy, color=bar_color, width=bar_width, yerr=errors, ecolor='#0c84c6', capsize=4)
    # plt.bar(x_positions, accuracy, color=bar_color, width=bar_width, yerr=errors, capsize=3)

    # 添加标题和轴标签
    plt.xlabel('Error Tolerance Rate (ETR)', fontsize=text_size, fontweight='bold')
    plt.ylabel('Accuracy (%)', color=bar_color, fontsize=text_size, fontweight='bold')  # 设置Y轴标签颜色
    plt.xticks(x_positions, rate, fontsize=ticks_size, fontweight='bold')
    plt.yticks(color=bar_color, fontsize=ticks_size, fontweight='bold')

    # 设置y轴的范围
    plt.ylim(70, 100)

    ax = plt.gca()
    # 为90到100之间的区域添加背景颜色
    # ax.axhspan(90, 100, facecolor='#E7DAD2', alpha=0.3, zorder=0.5)
    # 自定义网格线
    for y in range(70, 101, 5):
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  # 普通的线

    # 显示图形
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy.pdf", format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


# draw_accuracy()

def read_data(file_name):
    # 初始数据结构
    train_losses, test_losses = [], []
    train_groups, test_groups = [], []

    with open(file_name, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            epoch, train_loss, test_loss = int(parts[0]), float(parts[1]), float(parts[2])

            # 将train_loss和test_loss分别加入它们的数组
            train_losses.append(train_loss)
            test_losses.append(test_loss)

    # 根据0-250一组，将数据分组
    num_per_group = 251
    for i in range(0, len(train_losses), num_per_group):
        train_groups.append(train_losses[i:i + num_per_group])
        test_groups.append(test_losses[i:i + num_per_group])

    return train_groups, test_groups


# # 输出检查
# for i, (train_g, test_g) in enumerate(zip(train_groups, test_groups)):
#     print(f"Group {i}:")
#     print(f"Train Loss: {train_g}")
#     print(f"Test Loss: {test_g}")
#     print("-----------------------------")

def draw_loss():
    train_groups_per, test_groups_per = read_data('per_loss.txt')
    train_groups_con, test_groups_con = read_data('con_loss.txt')

    # 计算均值和方差
    epochs = np.arange(0, 250)
    epochs_shown = 200
    per_mean_losses, con_mean_losses = np.mean(test_groups_per, axis=0)[:epochs_shown], np.mean(test_groups_con,
                                                                                                axis=0)[:epochs_shown]
    per_var_losses, con_var_losses = np.var(test_groups_per, axis=0)[:epochs_shown], np.var(test_groups_per, axis=0)[
                                                                                     :epochs_shown]
    per_std_losses, con_std_losses = np.std(test_groups_per, axis=0)[:epochs_shown], np.std(test_groups_per, axis=0)[
                                                                                     :epochs_shown]
    epochs = epochs[:epochs_shown]
    yticks = [0, 0.5, 1, 1.5, 2, 2.5]
    # 创建图形
    per_color = '#ffa510'
    con_color = '#0c84c6'
    plt.figure(figsize=(15, 5))
    ######################Per###############################################
    plt.subplot(1, 2, 1)
    plt.plot(epochs, per_mean_losses, color=per_color, label='Average Loss of Per.')
    # plt.fill_between(epochs, per_mean_losses - per_var_losses, per_mean_losses + per_var_losses, color='blue', alpha=0.2,
    #                  label='Loss Fluctuation Range (Variance)')
    plt.fill_between(epochs, per_mean_losses - per_std_losses, per_mean_losses + per_std_losses, color=per_color,
                     alpha=0.3,
                     label='Loss Fluctuation Range')
    plt.xlabel('Number of Epoch', fontsize=text_size, fontweight='bold')
    plt.ylabel('Loss', color=per_color, fontsize=text_size, fontweight='bold')
    plt.xticks(fontsize=ticks_size, fontweight='bold')
    plt.yticks(yticks, fontsize=ticks_size, fontweight='bold', color=per_color)
    plt.legend(prop={'size': text_size, 'weight': 'bold'})
    plt.grid(True)
    plt.xlim(-10, epochs_shown + 10)  # 设置x轴范围
    plt.text(0.5, -0.22, '(a)', transform=plt.gca().transAxes, size=text_size)
    ######################Con##############################################
    plt.subplot(1, 2, 2)
    plt.plot(epochs, con_mean_losses, color=con_color, label='Average Loss of Con.')
    # plt.fill_between(epochs, con_mean_losses - con_var_losses, con_mean_losses + con_var_losses, color= con_color, alpha=0.3,
    #                                   label='Loss Fluctuation Range (Variance)')
    plt.fill_between(epochs, con_mean_losses - con_std_losses, con_mean_losses + con_std_losses, color=con_color,
                     alpha=0.3,
                     label='Loss Fluctuation Range')
    plt.xlabel('Number of Epoch', fontsize=text_size, fontweight='bold')
    plt.ylabel('Loss', color=con_color, fontsize=text_size, fontweight='bold')
    plt.xticks(fontsize=ticks_size, fontweight='bold')
    plt.yticks(yticks, fontsize=ticks_size, fontweight='bold', color=con_color)
    plt.legend(prop={'size': text_size, 'weight': 'bold'})
    plt.grid(True)
    plt.xlim(-10, epochs_shown + 10)  # 设置x轴范围
    plt.text(0.5, -0.22, '(b)', transform=plt.gca().transAxes, size=text_size)
    #######################################################################

    plt.tight_layout()
    plt.savefig("loss.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


# draw_loss()


def draw_dataset_volume():
    # 数据
    dataset_volume = ["5k", "7.5k", "10k", "12.5k", "15k", "17.5k", "20k", "22.5k"]
    per_accuracy = [68.45, 80.86, 79.34, 85.97, 86.46, 89.52, 90.52, 91]
    # per_error_range = [[67.11, 78.37, 79.34, 84.01, 84.31, 87.98, 89.44], [70.88, 80.98, 80.26, 86.06, 86.80, 89.52, 90.52]]
    per_error_range = [[1.34, 2.49, 0.5, 1.98, 2.15, 1.54, 1.08, 0.95],
                       [2.43, 0.63, 0.92, 0.49, 0.34, 0.5, 1, 0.39]]
    con_accuracy = [70.09, 73.99, 80.85, 80.79, 87.12, 91.32, 92.52, 93.08]
    con_error_range = [[0.94, 1.08, 0.3, 0.5, 0.15, 1.45, 0.7, 0.34],
                       [1.18, 2.63, 1.87, 1.49, 0.2, 0.9, 0.6, 0.27]]

    # 绘制折线图
    per_color = '#ffa510'
    con_color = '#0c84c6'
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.errorbar(dataset_volume, per_accuracy, color=per_color, linestyle='-', lw=2, fmt='o', markersize=3, label='Average Accuracy of Per.',
                 yerr=per_error_range, capsize=4, ecolor='r')
    plt.xlabel("Size of Dataset", fontsize=text_size, fontweight='bold')
    plt.ylabel("Accuracy (%)", color=per_color, fontsize=text_size, fontweight='bold')
    plt.xticks(fontsize=ticks_size, fontweight='bold')
    plt.yticks(fontsize=ticks_size, fontweight='bold', color=per_color)
    plt.legend(loc='lower right', prop={'size': text_size, 'weight': 'bold'})
    plt.ylim(65, 95)
    plt.grid(True)
    plt.text(0.5, -0.22, '(a)', transform=plt.gca().transAxes, size=text_size)

    plt.subplot(1,2,2)
    plt.errorbar(dataset_volume, con_accuracy, color=con_color, linestyle='-', lw=2, fmt='o', markersize=3, label='Average Accuracy of Con.',
                 yerr=con_error_range, capsize=4, ecolor='#32cd32')
    plt.xlabel("Size of Dataset", fontsize=text_size, fontweight='bold')
    plt.ylabel("Accuracy (%)", color=con_color, fontsize=text_size, fontweight='bold')
    plt.xticks(fontsize=ticks_size, fontweight='bold')
    plt.yticks(fontsize=ticks_size, fontweight='bold', color=con_color)
    plt.legend(prop={'size': text_size, 'weight': 'bold'})
    plt.ylim(65, 95)
    plt.grid(True)
    plt.text(0.5, -0.22, '(b)', transform=plt.gca().transAxes, size=text_size)

    # 显示图形
    plt.tight_layout()
    plt.savefig("data_size.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

draw_dataset_volume()

def create_matrix():
    # 创建一个 20 x 20 的全零矩阵
    matrix = np.zeros((20, 20), dtype=int)

    # 修改矩阵的特定区域的值
    matrix[2:13, 11:18] = 1
    matrix[3:8,2:8] = 1  # wifi
    # matrix[11:19, 1:8] = 1 # 5g

    # 创建图表
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(matrix, cmap=plt.cm.colors.ListedColormap(['lightgray', '#0c84c6']))
    # cax = ax.matshow(matrix, cmap=plt.cm.colors.ListedColormap(['lightgray', '#ffa510']))

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    # 展示图表
    plt.savefig('case_study_input_wifi.png', transparent=True)
    # plt.savefig('case_study_input_5g.png', transparent=True)
    plt.show()
    return matrix


def apply_affine_transformation(matrix):
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8))

    # 为了获得从左下角看的仿射变换
    trans = (Affine2D()
             .translate(-10, -10)  # 将中心移至原点
             .rotate_deg(-45)
             .scale(1,0.5)
             .translate(10, 10))  # 将中心移回原位置

    ax.imshow(matrix, cmap=plt.cm.colors.ListedColormap(['lightgray', '#0c84c6']), transform=trans + ax.transData)
    # ax.imshow(matrix, cmap=plt.cm.colors.ListedColormap(['lightgray', '#ffa510']), transform=trans + ax.transData)

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(-5, 25)  # 调整视图范围，确保整个图像可见
    # ax.set_ylim(-5, 30)  # 调整视图范围，确保整个图像可见

    # 保存图表
    plt.savefig('skewed_input_wifi.png', transparent=True, bbox_inches='tight', pad_inches=0)
    # plt.savefig('skewed_input_5g.png', transparent=True, bbox_inches='tight', pad_inches=0)

    # 展示图表
    plt.show()

# matrix = create_matrix()
# apply_affine_transformation(matrix)


def output_coverage(file, threshold):
    df = pd.read_csv(os.path.join("./", file), delimiter=' ', skiprows=3, header=None, usecols=[5])
    df['binary'] = np.where(df[5] >= threshold, 1, 0)
    matrix = df['binary'].values.reshape(20, 20)
    return matrix


def visualize_matrices(input_matrix, output_matrix, colors):
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8))

    # 画输入矩阵
    ax.imshow(input_matrix, cmap=plt.cm.colors.ListedColormap(['lightgray', colors[0]]), alpha=0.9)

    # 画输出矩阵
    ax.imshow(output_matrix, cmap=plt.cm.colors.ListedColormap(['lightgray', colors[1]]), alpha=0.5)

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    # 展示图表
    plt.show()


def apply_affine_transformation_and_show(input_matrix, output_matrix, colors):
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8))

    # 为了获得从左下角看的仿射变换
    trans = (Affine2D()
             .translate(-10, -10)  # 将中心移至原点
             .rotate_deg(-45)
             .scale(1, 0.5)
             .translate(10, 10))  # 将中心移回原位置

    # 画输入矩阵
    ax.imshow(input_matrix, cmap=plt.cm.colors.ListedColormap(['lightgray', colors[0]]),
              transform=trans + ax.transData, alpha=0.9)

    # 画输出矩阵
    ax.imshow(output_matrix, cmap=plt.cm.colors.ListedColormap(['lightgray', colors[1]]),
              transform=trans + ax.transData, alpha=0.5)

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(-5, 25)  # 调整视图范围，确保整个图像可见

    # 保存图表
    # plt.savefig('skewed_combined_wifi.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.savefig('skewed_combined.png', transparent=True, bbox_inches='tight', pad_inches=0)

    # 展示图表
    plt.show()

# 示例：
input_matrix_5g, input_matrix_wifi = np.zeros((20, 20)), np.zeros((20, 20))
input_matrix_wifi[2:13, 11:18] = 1
input_matrix_wifi[3:8,2:8] = 1
# input_matrix_5g[11:19, 1:8] = 1
#
# output_matrix_wifi = output_coverage("1.000e-03_5.00_wifi.p2m", -97)
# output_matrix_5g = output_coverage("1.000e-03_5.00_5g.p2m", -78)
#
colors_wifi = ["#0c84c6", "#7cd6cf"]
# colors_5g = ['#ffa510', 'lightyellow']

# visualize_matrices(input_matrix_wifi, output_matrix_wifi, colors_wifi)
# apply_affine_transformation_and_show(input_matrix_wifi, output_matrix_wifi, colors_wifi)
# visualize_matrices(input_matrix_5g, output_matrix_5g, colors_5g)
# apply_affine_transformation_and_show(input_matrix_5g, output_matrix_5g, colors_5g)





