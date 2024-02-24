import glob
import os

import numpy as np



def remove_redundant_data(data_2d, inputpath, interval=0.1, max_points_per_grid=1, output=None):
    a = np.load(data_2d)
    # 设置网格间隔和每个网格中最多取的数据点数
    # 获取数据的范围
    x_min = np.min(a[:, 0])  # 假设 x 数据在第一列
    x_max = np.max(a[:, 0])
    y_min = np.min(a[:, 1])  # 假设 y 数据在第二列
    y_max = np.max(a[:, 1])

    # 根据间隔划分网格
    x_bins = np.arange(x_min, x_max + interval, interval)
    y_bins = np.arange(y_min, y_max + interval, interval)
    # print(len(x_bins), len(y_bins))
    # 划分网格并取数据
    grid_data = []

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            x_min_bin = x_bins[i]
            x_max_bin = x_bins[i + 1]
            y_min_bin = y_bins[j]
            y_max_bin = y_bins[j + 1]

            # 获取每个网格内的数据索引
            indices_in_grid = np.where(
                (a[:, 0] >= x_min_bin) & (a[:, 0] < x_max_bin) &
                (a[:, 1] >= y_min_bin) & (a[:, 1] < y_max_bin)
            )[0]

            # 根据需要采取规则
            if len(indices_in_grid) > 0:
                # 获取最多指定数量的数据点的索引
                if len(indices_in_grid) > max_points_per_grid:
                    np.random.shuffle(indices_in_grid)  # 随机打乱顺序
                    indices_in_grid = indices_in_grid[:max_points_per_grid]

                # 保存单个网格中的索引到整体列表中
                grid_data.append(indices_in_grid.tolist())  # 转换为 Python 列表并添加

    # 将 grid_data 转换为平铺的单个索引列表
    flat_indices = np.concatenate(grid_data)
    # print(len(flat_indices))
    # np.save('indices.npy',flat_indices)
    # plotscatter(plotflag, a, flat_indices, x_bins, y_bins)

    if output is None:
        path_info = os.path.split(data_2d)
        directory = path_info[0]  # 文件所在目录
        file_name = path_info[1]  # 文件名

        # 从文件名中分割出指定部分
        file_name_parts = file_name.split('_')
        part_1 = file_name_parts[3]  # 获取 '202401081123'
        part_2 = '_'.join(file_name_parts[4:7])  # 获取 'trainf_5_tsne'
        # part_3 = file_name_parts[-1].split('.')[0]  # 获取 '2'（去除扩展名）
        name2 = f'output_{part_1}_{part_2}_{interval}_{max_points_per_grid}_{len(flat_indices)}.data'
        output = os.path.join(directory, name2)
    print('筛选前数量：', len(a))
    saveoutput(flat_indices, inputpath, output)

def saveoutput(flat_indices, inputpath, output):
    indexarr = flat_indices
    begin_found = False
    index = 0

    with open(inputpath, 'r', newline='\n') as input_f, open(output, 'w', newline='\n') as output_f:
        for line in input_f:
            if begin_found:
                if line.strip() == 'end':
                    begin_found = False
                    if index in indexarr:
                        output_f.write(line)
                    index += 1
                else:
                    if index in indexarr:
                        output_f.write(line)
            elif line.strip() == 'begin':
                if index in indexarr:
                    output_f.write(line)
                begin_found = True

    print('筛选后数量：', len(flat_indices))
    print('输出数据:', output)