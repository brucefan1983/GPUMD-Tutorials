import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def load_thermal_data(direction, base_dir="."):
    """load"""
    file_path = os.path.join(base_dir, f"{direction}/kappa_J_J_all_combined.txt")
    
    try:
        data = np.loadtxt(file_path)
        
        if data.ndim < 2 or data.shape[1] < 3:
            raise ValueError(f"数据格式不正确: {file_path}，需要至少三列数据")
            
        time = data[:, 0]
        value = data[:, 1:]
        
        return time, value
            
    except Exception as e:
        print(f"无法加载数据 {file_path}: {e}")
        return None, None

def calculate_average(x_value, y_value, z_value):
    """计算三个方向的平均值和误差"""
    stacked = np.hstack([x_value, y_value, z_value])
    n_total = stacked.shape[1] # 2000*600
    print(n_total)
    avg = np.mean(stacked, axis=1)

    std = np.std(stacked, axis=1, ddof=1) /np.sqrt(n_total) # ddof=1表示样本标准差
    # 计算平均值
    print(avg) 
    
    return avg, std

def calculate_time_average(time, value, errors, time_range=(1.0, 3.0)):
    """
    计算三个方向在指定时间范围内的平均值和误差

    参数:
    value, error: 各方向的热导率误差数组
    time_range: 时间范围的元组，格式 (起始时间, 终止时间)
    """
    start_time, end_time = time_range

    # 筛选在指定时间范围内的数据索引
    mask = (time >= start_time) & (time <= end_time)

    # 提取在时间范围内的数据
    value_in_range = value[mask]
    error_in_range = errors[mask]

    # 计算平均值（在时间范围内）
    average = (np.sum(value_in_range)) / (
            len(value_in_range))

    # 计算误差（假设三个方向独立，误差平方和开根号）
    combined_error = np.sqrt(np.sum(error_in_range ** 2) ) / (len(value_in_range))

    return average, combined_error

def plot_thermal_conductivity(x_dir="x", y_dir="y", z_dir="z", base_dir="."):
    """绘制三个方向的热导率对比图及平均值"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 加载各方向数据
    directions = [x_dir, y_dir, z_dir]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    labels = ['x', 'y', 'z']
    
    all_times = []
    all_values = []
    all_errors = []
    
    for dir_name, color, label in zip(directions, colors, labels):
        time, value = load_thermal_data(dir_name, base_dir)
        if time is not None:
            all_times.append(time)
            all_values.append(value)
            
            # 绘制热导率曲线
            #ax.plot(time, value, '-', color='gray', label=label)
            
            
            # 绘制误差范围
            # ax.fill_between(time, value - error, value + error, color=color, alpha=0.1)
    
    # 检查是否成功加载了所有三个方向的数据
    if len(all_times) == 3:
        # 确保时间轴一致（假设相同，否则需要插值处理）
        if np.allclose(all_times[0], all_times[1]) and np.allclose(all_times[0], all_times[2]):
            # 计算平均值和误差
            average, std = calculate_average(
                all_values[0], all_values[1], all_values[2]
            )
            #print(average) 
            # 绘制平均值曲线（黑色加粗）
            ax.plot(all_times[0], average, '-', color='red', linewidth=2, label='ave')
            ax.plot(all_times[0], average+std, '-', color='blue', linewidth=2, label='std')
            ax.plot(all_times[0], average-std, '-', color='blue', linewidth=2, label='std')
            #截取制定时间段内求平均
            #average_final, avg_error_final = calculate_time_average(
            #    time, average, avg_error,
            #    time_range=(1.0, 3.0)
            #)
            #end_time=4
            #ax.plot([0.0, end_time], [average_final, average_final], '--', linewidth=2, label='average(1-3ps)')
            #ax.annotate(f'平均值: {average_final:.6f} ± {avg_error_final:.6f}', xy=(2.0, average), xytext=(2.0, average + 0.1),
            #            arrowprops=dict(facecolor='black', arrowstyle='->'))
            #ax.plot([0.0, end_time], [0.4532, 0.4532], '--', linewidth=2, color='#e74c3c', label='Cepstral analysis')
            #ax.fill_between([0.0, end_time], 0.4532-0.066713, 0.4532+0.066713, color='#e74c3c', alpha=0.3)

            #print(f"Average: {average_final:.6f}± {avg_error_final:.6e} W/m/K")

            # 绘制平均误差范围
            #ax.fill_between(all_times[0], average - avg_error, average + avg_error, 
            #               color='black', alpha=0.1, label='error range')
        else:
            print("警告：三个方向的时间轴不一致，无法计算平均值")
    
    # 设置图表属性
    ax.set_xlabel('time (ps)')
    ax.set_ylabel('L00 (W/mK)')
    #ax.grid(True, linestyle='--', alpha=0.7)
    #ax.legend()
    #plt.ylim([0e-8, 12e-8])
    plt.tick_params(direction='in')
    #plt.xlim([0,end_time])
    plt.savefig('nepC-650K-kkp.eps')
    # 保存图表
    plt.show()

if __name__ == "__main__":
    # 默认从当前目录下的x、y、z文件夹读取数据
    plot_thermal_conductivity()
