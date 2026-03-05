import numpy as np
import matplotlib.pyplot as plt
import os


def load_hac(base_dir="."):
    """加载热导率数据"""
    file_path = os.path.join(base_dir, "hac.out")  # 修正文件路径
    
    try:
        # 读取数据，假设至少三列
        data = np.loadtxt(file_path)
        print(f"成功加载数据，形状: {data.shape}")
        
            
        time = data[:, 0]      # 第一列: 时间
        value = data[:, 10]    # 第11列: kz值
        return time, value
            
    except Exception as e:
        print(f"无法加载数据 {file_path}: {e}")
        return None, None

def load_ourGK(base_dir="."):
    """加载热导率数据"""
    file_path = os.path.join(base_dir, "kappa_ee.txt")  # 修正文件路径

    try:
        # 读取数据，假设至少三列
        data = np.loadtxt(file_path)


        time = data[:, 0]      # 第1列: 时间
        value = data[:, 1]    # 第2列: kz值
        return time, value

    except Exception as e:
        print(f"无法加载数据 {file_path}: {e}")
        return None, None



def plot_thermal_conductivity(base_dir="."):
    """绘制热导率曲线"""
    fig, ax = plt.subplots(figsize=(12, 8))  # 修正括号
    
    time, value_from_gpumd = load_hac(base_dir)    
    # 绘制热导率曲线
    ax.plot(time, value_from_gpumd, 'b-', linewidth=2, label='GPUMD')  # 指定颜色和线宽
    print(np.mean(value_from_gpumd[-int(len(value_from_gpumd)/2):]))

    time, value_from_ourGK = load_ourGK(base_dir)
    print(np.mean(value_from_ourGK[-int(len(value_from_gpumd)/2):]))
    # 绘制热导率曲线
    ax.plot(time, value_from_ourGK, 'r-', linewidth=2, label='OurGK')  # 指定颜色和线宽

    ratio = value_from_gpumd/value_from_ourGK
    #print(ratio)
    #ax.plot(time, ratio,label='GPUMD/ourGK')
    # 设置图表属性
    ax.set_xlabel('time (ps)')
    ax.set_ylabel('kz (W/mK)')
    #plt.ylim(0.2, 2)
    plt.tick_params(direction='in')
    #ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.savefig("GK_comparison.eps")    
    plt.show()

if __name__ == "__main__":
    # 默认从当前目录读取数据
    plot_thermal_conductivity()
