import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from ase.io import read
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor

# 读取热流数据
def read_heat_flux_data(file_path):
    data = np.loadtxt(file_path, comments='#')
    return data

# 读取初始结构文件并计算体积
def calculate_volume(xyz_file):
    atoms = read(xyz_file)
    return atoms.get_volume()

# 计算自相关函数
def autocorrelation(data, dt, max_lag):
    n = len(data)
    #data -= np.mean(data)
    result = np.correlate(data, data, mode='full')[n-1:] / (n - np.arange(n))
    return np.arange(len(result[:max_lag])) * dt, result[:max_lag]

# 计算互相关函数
def cross_correlation(data1, data2, dt, max_lag):
    n = len(data1)
    #data1 -= np.mean(data1)
    #data2 -= np.mean(data2)
    result = np.correlate(data1, data2, mode='full')[n-1:] / (n - np.arange(n))
    return np.arange(len(result[:max_lag])) * dt, result[:max_lag]

# 计算Green-Kubo热导率和累积热导率
def calculate_kappa(J, v, V, T, dt, max_lag, direction):
    k_B, A_to_m = 1.380649e-23, 1e-10
    V = V * (A_to_m**3)
    #
    J_time, J_corr = autocorrelation(J[:, direction], dt, max_lag)
    v_time, v_corr = autocorrelation(v[:, direction], dt, max_lag)

    cross_time, cross_corr = cross_correlation(J[:, direction], v[:, direction], dt, max_lag)

    #
    kappa_ee = (1 / (k_B * V * T**2)) * cumtrapz(J_corr, J_time, initial=0)
    kappa_ev = (1 / (k_B * V * T**2)) * cumtrapz(cross_corr, J_time, initial=0)
    kappa_vv = (1 / (k_B * V * T**2)) * cumtrapz(v_corr, v_time, initial=0)
    #
    return J_time, J_corr, v_corr, cross_corr, kappa_ee, kappa_ev, kappa_vv 

# 使用最后1/4数据计算数值平均值和标准误差
def calculate_avg_and_error(data):
    quarter_length = len(data[0]) // 2
    avg_values = [np.mean(d[-quarter_length:]) for d in data]
    avg = np.mean(avg_values)
    error = np.std(avg_values) / np.sqrt(len(avg_values))
    return avg, error, avg_values
# 绘图函数
def plot_correlation_and_save(time, data_list, mean_data, title, filename):
    plt.figure(figsize=(4*0.9, 3*0.9))
    #for data in data_list:
    #    plt.plot(time * 1e12, data, color='grey', alpha=0.5)
    plt.semilogy(time * 1e12, mean_data, color='black', linewidth=2, label='Mean')
    plt.xlabel('Time (ps)')
    plt.ylabel('Correlation')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
# 绘图函数
def plot_kee(time, data_list, mean_data, std_data, title, filename):
    plt.figure(figsize=(4*0.9, 3*0.9))
    for data in data_list:
        plt.plot(time * 1e12, data, color='grey', alpha=0.5)
    plt.plot(time * 1e12, mean_data, color='black', linewidth=2, label='Mean')
    plt.plot(time * 1e12, mean_data+std_data, color='red', linewidth=1, label='Error', linestyle='--')
    plt.plot(time * 1e12, mean_data-std_data, color='red', linewidth=1, label='Error', linestyle='--')
    plt.xlabel('Time (ps)')
    plt.ylabel(r'$\kappa$ (W/mK)')
    plt.ylim([0.2,0.8])
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 绘图函数
def plot_kev(time, data_list, mean_data, std_data, title, filename):
    plt.figure(figsize=(4*0.9, 3*0.9))
    for data in data_list:
        plt.plot(time * 1e12, data, color='grey', alpha=0.5)
    plt.plot(time * 1e12, mean_data, color='black', linewidth=2, label='Mean')
    plt.plot(time * 1e12, mean_data+std_data, color='red', linewidth=1, label='Error', linestyle='--')
    plt.plot(time * 1e12, mean_data-std_data, color='red', linewidth=1, label='Error', linestyle='--')
    plt.xlabel('Time (ps)')
    plt.ylabel(r'$\kappa$')
    plt.ylim([1e-8,3e-8])
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 绘图函数
def plot_kvv(time, data_list, mean_data, std_data, title, filename):
    plt.figure(figsize=(4*0.9, 3*0.9))
    for data in data_list:
        plt.plot(time * 1e12, data, color='grey', alpha=0.5)
    plt.plot(time * 1e12, mean_data, color='black', linewidth=2, label='Mean')
    plt.plot(time * 1e12, mean_data+std_data, color='red', linewidth=1, label='Error', linestyle='--')
    plt.plot(time * 1e12, mean_data-std_data, color='red', linewidth=1, label='Error', linestyle='--')
    plt.xlabel('Time (ps)')
    plt.ylabel(r'$\kappa$')
    plt.ylim([1e-15,3e-15])
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def calculate_y(a, b, c):
    """y = a - b²/c"""
    return a - b**2 / c

def calculate_y_error(a, b, c, da, db, dc):
    """
    y = a - b²/c

    """
    dyda = np.ones_like(a)  # 偏导数 ∂y/∂a = 1
    dydb = -2 * b / c       # 偏导数 ∂y/∂b = -2b/c
    dydc = b**2 / c**2      # 偏导数 ∂y/∂c = b²/c²

    dy = np.sqrt((da * dyda)**2 + (db * dydb)**2 + (dc * dydc)**2)

    return dy

def Extract_heatflux(prod_dir): 
    """
    提取热流数据并保存到文本文件中。

    """
    compute = {}
    compute = pd.read_csv(
        f"{prod_dir}/compute.out",
        sep=r"\s+",
        names=[
                        "jv1[1]",
                        "jv2[1]",
                        "jv3[1]",
                        "jv1[2]",
                        "jv2[2]",
                        "jv3[2]",
                        "jv1[3]",
                        "jv2[3]",
                        "jv3[3]",
                        "jk1[1]",
                        "jk2[1]",
                        "jk3[1]",
                        "jk1[2]",
                        "jk2[2]",
                        "jk3[2]",
                        "jk1[3]",
                        "jk2[3]",
                        "jk3[3]",
                        "v1[1]",
                        "v2[1]",
                        "v3[1]",
                        "v1[2]",
                        "v2[2]",
                        "v3[2]",
                        "v1[3]",
                        "v2[3]",
                        "v3[3]",
                    ],
                )
    # 计算总热流 j[1], j[2], j[3]
    for ii in range(1, 4):
        compute[f"j[{ii}]"] = (
                        compute[f"jk1[{ii}]"]
                        + compute[f"jv1[{ii}]"]
                        + compute[f"jk2[{ii}]"]
                        + compute[f"jv2[{ii}]"]
                        + compute[f"jk3[{ii}]"]
                        + compute[f"jv3[{ii}]"]
                    )
    # 计算总导热流 jv[1], jv[2], jv[3]
    for ii in range(1, 4):
        compute[f"jv[{ii}]"] = (
                         compute[f"jv1[{ii}]"]
                        + compute[f"jv2[{ii}]"]
                        + compute[f"jv3[{ii}]"]
                    )
    # 计算总对流热流 jk[1], jk[2], jk[3]
    for ii in range(1, 4):
        compute[f"jk[{ii}]"] = (
                        compute[f"jk1[{ii}]"]
                        + compute[f"jk2[{ii}]"]
                        + compute[f"jk3[{ii}]"]
                    )
    # 提取 j[1], j[2], j[3] 数据
    j_data = compute[[f"j[{i}]" for i in range(1, 4)]].to_numpy()

    # 提取 jv[1], jv[2], jv[3] 数据
    jv_data = compute[[f"jv[{i}]" for i in range(1, 4)]].to_numpy()

    # 提取 jk[1], jk[2], jk[3] 数据
    jk_data = compute[[f"jk[{i}]" for i in range(1, 4)]].to_numpy()

    v1_data = compute[[f"v1[{i}]" for i in range(1, 4)]].to_numpy()

    # save
    np.savetxt(
            f"{prod_dir}/j.txt",
            j_data,
            header="j[1] j[2] j[3]",
            fmt="%.6e",
            delimiter=" "
        )
    np.savetxt(
            f"{prod_dir}/jv.txt",
            jv_data,
            header="jv[1] jv[2] jv[3]",
            fmt="%.6e",
            delimiter=" "
        )
    np.savetxt(
            f"{prod_dir}/jk.txt",
            jk_data,
            header="jk[1] jk[2] jk[3]",
            fmt="%.6e",
            delimiter=" "
        )
    np.savetxt(
            f"{prod_dir}/v1.txt",
            v1_data,
            header="v1[1] v1[2] v1[3]",
            fmt="%.6e",
            delimiter=" "
        )



# 主函数
def main(base_dir, xyz_file, T, dt, direction):
    JJ_corr_all, vv_corr_all, Jv_corr_all = [], [], []
    kappa_ee_all, kappa_vv_all, kappa_ev_all = [], [], []
    volume = calculate_volume(xyz_file)

    Extract_heatflux(base_dir)

    chunk_size = 200000
    j = read_heat_flux_data('jv.txt')
    v1= read_heat_flux_data('v1.txt')
    eV_to_J, AMU_to_kg = 1.60218e-19, 1.66054e-27
    j = j * (eV_to_J**1.5) / (AMU_to_kg**0.5)
    v1 = v1 * (eV_to_J**0.5) * (AMU_to_kg**0.5)
    
    total_data=len(j)

    # 关联时: ps
    t_correlation= 10
    max_lag = round((t_correlation*1e-12)/dt)
    print(max_lag)
    # step, each step=5fs
    #
    for start_idx in range(0, total_data, chunk_size):
        end_idx = min(start_idx + chunk_size, total_data)
        print(f"Processing data chunk: {start_idx} to {end_idx}")
        j_chunk = j[start_idx : end_idx]
        v1_chunk = v1[start_idx : end_idx]

        [J_time, JJ_corr, vv_corr, Jv_corr, kappa_ee, kappa_ev, kappa_vv] = calculate_kappa(j_chunk, v1_chunk, volume, T, dt, max_lag, direction)
        JJ_corr_all.append(JJ_corr)
        vv_corr_all.append(vv_corr)
        Jv_corr_all.append(Jv_corr)
        kappa_ee_all.append(kappa_ee)
        kappa_vv_all.append(kappa_vv)
        kappa_ev_all.append(kappa_ev)


    JJ_corr_mean = np.mean(JJ_corr_all, axis=0)
    vv_corr_mean = np.mean(vv_corr_all, axis=0)
    Jv_corr_mean = np.mean(Jv_corr_all, axis=0)
    kappa_ee_mean = np.mean(kappa_ee_all, axis=0)
    kappa_vv_mean = np.mean(kappa_vv_all, axis=0)
    kappa_ev_mean = np.mean(kappa_ev_all, axis=0)
    kappa_ee_error = np.std(kappa_ee_all, axis=0) / np.sqrt(len(kappa_ee_all)-1)

    # 
    np.savetxt('kappa_ee.txt', 
           np.column_stack([J_time*1e12, kappa_ee_mean, kappa_ee_mean+kappa_ee_error, kappa_ee_mean-kappa_ee_error]),
           fmt='%.6e', 
           delimiter='\t', 
           header='time(ps) \tkappa_ee_mean\tkappa_ee_upper\tkappa_ee_lower')



if __name__ == "__main__":
    # direction = x y z
    direction = 2
    main(".", "model.xyz", 650, 1e-15, direction)
