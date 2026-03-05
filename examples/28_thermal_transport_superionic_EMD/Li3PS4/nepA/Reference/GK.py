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
def calculate_kappa(J,Jv, Jk, v, V, T, dt, max_lag, direction):
    """
    J = Jv + Jk
    """
    k_B, A_to_m = 1.380649e-23, 1e-10
    V = V * (A_to_m**3)
    #
    J_J_time, J_J_corr = autocorrelation(J[:, direction], dt, max_lag)
    v_v_time, v_v_corr = autocorrelation(v[:, direction], dt, max_lag)
    Jv_Jv_time, Jv_Jv_corr = autocorrelation(Jv[:, direction], dt, max_lag)
    Jk_Jk_time, Jk_Jk_corr = autocorrelation(Jk[:, direction], dt, max_lag)

    J_v_time, J_v_corr = cross_correlation(J[:, direction], v[:, direction], dt, max_lag)
    Jv_Jk_time, Jv_Jk_corr = cross_correlation(Jv[:, direction], Jk[:, direction], dt, max_lag)
    #
    kappa_J_J = (1 / (k_B * V * T**2)) * cumtrapz(J_J_corr, J_J_time, initial=0)
    kappa_Jv_Jv = (1 / (k_B * V * T**2)) * cumtrapz(Jv_Jv_corr, Jv_Jv_time, initial=0)
    kappa_Jk_Jk = (1 / (k_B * V * T**2)) * cumtrapz(Jk_Jk_corr, Jk_Jk_time, initial=0)
    kappa_Jv_Jk = (1 / (k_B * V * T**2)) * cumtrapz(Jv_Jk_corr, Jv_Jk_time, initial=0)
    # kappa_Jv_Jk中有一个2的系数，在后面处理

    kappa_J_v = (1 / (k_B * V * T**2)) * cumtrapz(J_v_corr, J_v_time, initial=0)
    kappa_v_v = (1 / (k_B * V * T**2)) * cumtrapz(v_v_corr, v_v_time, initial=0)
    #
    return J_J_time, kappa_J_J, kappa_Jv_Jv, kappa_Jk_Jk, kappa_Jv_Jk, kappa_J_v, kappa_v_v

# 使用最后1/2数据计算数值平均值和标准误差
def calculate_avg_and_error(data, data_error):
    length = int(len(data) / 2)
    avg_value = np.mean(data[-length:])
    error = np.mean(data_error[-length:])
    return avg_value, error

def calculate_y(a, b, c):
    """y = a - b²/c """
    return a - b**2 / c

def calculate_y_error(a, b, c, da, db, dc):
    """
    Error of y = a - b²/c

    Parameter:
        a, b, c
        da, db, dc

    Returen:
        dy
    """
    # 
    dyda = np.ones_like(a)  # ∂y/∂a = 1
    dydb = -2 * b / c       # ∂y/∂b = -2b/c
    dydc = b**2 / c**2      # ∂y/∂c = b²/c²

    # 
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


def main(base_dir, xyz_file, T, dt, direction):

    chunk_size = 20000

    # 提取j,jv,jk, v1
    volume = calculate_volume(os.path.join(base_dir,xyz_file))
    Extract_heatflux(base_dir)

    # Read flux file
    j = read_heat_flux_data(os.path.join(base_dir, 'j.txt'))
    jv = read_heat_flux_data(os.path.join(base_dir, 'jv.txt'))
    jk = read_heat_flux_data(os.path.join(base_dir, 'jk.txt'))
    v1= read_heat_flux_data(os.path.join(base_dir,'v1.txt'))


    # Conversion 
    eV_to_J, AMU_to_kg = 1.60218e-19, 1.66054e-27
    j = j * (eV_to_J**1.5) / (AMU_to_kg**0.5)
    jv = jv * (eV_to_J**1.5) / (AMU_to_kg**0.5)
    jk = jk * (eV_to_J**1.5) / (AMU_to_kg**0.5)
    v1 = v1 * (eV_to_J**0.5) * (AMU_to_kg**0.5)
    
    total_data=len(j)

    # correlation time: ps
    t_correlation= 10
    max_lag = round((t_correlation*1e-12)/dt)
    print(max_lag)

    kappa_J_J_all, kappa_Jv_Jv_all, kappa_Jk_Jk_all, kappa_Jv_Jk_all, kappa_J_v_all, kappa_v_v_all=[], [], [], [], [], []
    # 开始分段计算GK
    for start_idx in range(0, total_data, chunk_size):
        end_idx = min(start_idx + chunk_size, total_data)
        j_chunk = j[start_idx : end_idx]
        jv_chunk = jv[start_idx : end_idx]
        jk_chunk = jk[start_idx : end_idx]
        v1_chunk = v1[start_idx : end_idx]

        [J_J_time, kappa_J_J, kappa_Jv_Jv, kappa_Jk_Jk, kappa_Jv_Jk, kappa_J_v, kappa_v_v] = calculate_kappa(j_chunk, jv_chunk, jk_chunk, v1_chunk, volume, T, dt, max_lag, direction)
        kappa_J_J_all.append(kappa_J_J)
        kappa_Jv_Jv_all.append(kappa_Jv_Jv)
        kappa_Jk_Jk_all.append(kappa_Jk_Jk)
        kappa_Jv_Jk_all.append(kappa_Jv_Jk)
        kappa_J_v_all.append(kappa_J_v)
        kappa_v_v_all.append(kappa_v_v)
    print(f"Processing data chunk done")

    kappa_J_J_mean = np.mean(kappa_J_J_all, axis=0)
    kappa_Jv_Jv_mean = np.mean(kappa_Jv_Jv_all, axis=0)
    kappa_Jk_Jk_mean = np.mean(kappa_Jk_Jk_all, axis=0)
    kappa_Jv_Jk_mean = np.mean(kappa_Jv_Jk_all, axis=0)
    kappa_J_v_mean = np.mean(kappa_J_v_all, axis=0)
    kappa_v_v_mean = np.mean(kappa_v_v_all, axis=0)


    print(len(kappa_J_J_all))
    # calculate the standard deviation
    kappa_J_J_error = np.std(kappa_J_J_all, axis=0) / np.sqrt(len(kappa_J_J_all)-1)
    kappa_Jv_Jv_error = np.std(kappa_Jv_Jv_all, axis=0) / np.sqrt(len(kappa_Jv_Jv_all)-1)
    kappa_Jk_Jk_error = np.std(kappa_Jk_Jk_all, axis=0) / np.sqrt(len(kappa_Jk_Jk_all)-1)
    kappa_Jv_Jk_error = np.std(kappa_Jv_Jk_all, axis=0) / np.sqrt(len(kappa_Jv_Jk_all)-1)
    kappa_J_v_error = np.std(kappa_J_v_all, axis=0) / np.sqrt(len(kappa_J_v_all)-1)
    kappa_v_v_error = np.std(kappa_v_v_all, axis=0) / np.sqrt(len(kappa_v_v_all)-1)

    direction_map = {0: 'x', 1: 'y', 2: 'z'}
    dir_name = direction_map[direction]
    output_dir = os.path.join(base_dir, f"{dir_name}")
    os.makedirs(output_dir, exist_ok=True)
    def save_data(output_dir, time, data_list, prefix):
        # 
        combined_data = np.column_stack([time] + data_list)
        #combined_data = np.column_stack(data_list)
        #headers="Traj"
        np.savetxt(f"{output_dir}/{prefix}_combined.txt",
              combined_data,
              fmt='%.6e',  
              delimiter='\t',  
              comments='')  #

    # save original data
    save_data(output_dir, J_J_time*1e12, kappa_J_J_all, 'kappa_J_J_all')
    save_data(output_dir, J_J_time*1e12, kappa_Jv_Jv_all, 'kappa_Jv_Jv_all')
    save_data(output_dir, J_J_time*1e12, kappa_Jk_Jk_all, 'kappa_Jk_Jk_all')
    save_data(output_dir, J_J_time*1e12, kappa_Jv_Jk_all, 'kappa_Jv_Jk_all')
    save_data(output_dir, J_J_time*1e12, kappa_J_v_all, 'kappa_J_v_all')
    save_data(output_dir, J_J_time*1e12, kappa_v_v_all, 'kappa_v_v_all')

    # save the mean and error bar
    np.savetxt(os.path.join(output_dir,'kappa_J_J.txt'),
           np.column_stack([J_J_time*1e12, kappa_J_J_mean, kappa_J_J_error]),
           fmt='%.6e',
           delimiter='\t',
           header='time(ps) \tkappa_J_J_mean\tkappa_J_J_error')
    np.savetxt(os.path.join(output_dir,'kappa_Jv_Jv.txt'),
           np.column_stack([J_J_time*1e12, kappa_Jv_Jv_mean, kappa_Jv_Jv_error]),
           fmt='%.6e',
           delimiter='\t',
           header='time(ps) \tkappa_Jv_Jv_mean\tkappa_Jv_Jv_error')
    
    np.savetxt(os.path.join(output_dir,'kappa_Jk_Jk.txt'),
           np.column_stack([J_J_time*1e12, kappa_Jk_Jk_mean, kappa_Jk_Jk_error]),
           fmt='%.6e',
           delimiter='\t',
           header='time(ps) \tkappa_Jk_Jk_mean\tkappa_Jk_Jk_error')

    np.savetxt(os.path.join(output_dir,'kappa_Jv_Jk.txt'),
           np.column_stack([J_J_time*1e12, kappa_Jv_Jk_mean, kappa_Jv_Jk_error]),
           fmt='%.6e',
           delimiter='\t',
           header='time(ps) \tkappa_Jv_Jk_mean\tkappa_Jv_Jk_error')
    np.savetxt(os.path.join(output_dir,'kappa_J_v.txt'),
           np.column_stack([J_J_time*1e12, kappa_J_v_mean, kappa_J_v_error]),
           fmt='%.6e',
           delimiter='\t',
           header='time(ps) \tkappa_J_v_mean\tkappa_J_v_error')

    np.savetxt(os.path.join(output_dir,'kappa_v_v.txt'),
           np.column_stack([J_J_time*1e12, kappa_v_v_mean, kappa_v_v_error]),
           fmt='%.6e',
           delimiter='\t',
           header='time(ps) \tkappa_v_v_mean\tkappa_v_v_error')

    # 计算最后的平均值
    kappa_J_J_avg, kappa_J_J_avg_error = calculate_avg_and_error(kappa_J_J_mean, kappa_J_J_error)
    kappa_Jv_Jv_avg, kappa_Jv_Jv_avg_error = calculate_avg_and_error(kappa_Jv_Jv_mean, kappa_Jv_Jv_error)
    kappa_Jk_Jk_avg, kappa_Jk_Jk_avg_error = calculate_avg_and_error(kappa_Jk_Jk_mean, kappa_Jk_Jk_error)
    kappa_Jv_Jk_avg, kappa_Jv_Jk_avg_error = calculate_avg_and_error(kappa_Jv_Jk_mean, kappa_Jv_Jk_error)
    kappa_J_v_avg, kappa_J_v_avg_error = calculate_avg_and_error(kappa_J_v_mean, kappa_J_v_error)
    kappa_v_v_avg, kappa_v_v_avg_error = calculate_avg_and_error(kappa_v_v_mean, kappa_v_v_error)

    # Onsager correction
    kappa_total = calculate_y(kappa_J_J_avg, kappa_J_v_avg, kappa_v_v_avg)
    kappa_error = calculate_y_error(kappa_J_J_avg, kappa_J_v_avg, kappa_v_v_avg, 
            kappa_J_J_avg_error,kappa_J_v_avg_error,kappa_v_v_avg_error)

    
    return kappa_J_J_avg, kappa_Jv_Jv_avg, kappa_Jk_Jk_avg, kappa_Jv_Jk_avg, kappa_J_v_avg, kappa_v_v_avg, kappa_total, kappa_J_J_avg_error, kappa_Jv_Jv_avg_error, kappa_Jk_Jk_avg_error, kappa_Jv_Jk_avg_error, kappa_J_v_avg_error, kappa_v_v_avg_error, kappa_error


if __name__ == "__main__":
    folder_names = [ "650K"]
    time_step=5e-15
    with open("Temperature_kappa_650K.txt", 'w') as file:
                # 写入表头
        file.write("Temperature kappa_J_J_avg±error, kappa_Jv_Jv_avg±error, kappa_Jk_Jk_avg±error, kappa_Jv_Jk_avg±error, kappa_J_v_avg±error, kappa_v_v_avg±error, kappa_total±error\n")
        for folder in folder_names:
            temperature = int(folder.replace("K", ""))
            folder_path = os.path.join(".", folder)

            kappa_J_J_direc_all, kappa_Jv_Jv_direc_all, kappa_Jk_Jk_direc_all, kappa_Jv_Jk_direc_all, kappa_J_v_direc_all, kappa_v_v_direc_all = [], [], [], [], [], []
            kappa_total_direc_all = []
            kappa_J_J_direc_all_error, kappa_Jv_Jv_direc_all_error, kappa_Jk_Jk_direc_all_error, kappa_Jv_Jk_direc_all_error, kappa_J_v_direc_all_error, kappa_v_v_direc_all_error = [], [], [], [], [], []
            kappa_total_direc_all_error = []

            if not os.path.isdir(folder_path):
                print(f"文件 {folder_path} 不存在，跳过该文件。")
                continue

            for direction in [0, 1, 2]:
                kappa_J_J_avg, kappa_Jv_Jv_avg, kappa_Jk_Jk_avg, kappa_Jv_Jk_avg, kappa_J_v_avg, kappa_v_v_avg, kappa_total, kappa_J_J_avg_error, kappa_Jv_Jv_avg_error, kappa_Jk_Jk_avg_error, kappa_Jv_Jk_avg_error, kappa_J_v_avg_error, kappa_v_v_avg_error,kappa_error = main(folder_path, "restart.xyz", temperature, time_step, direction)
                print(f"方向 {direction} 的计算完成")
                kappa_J_J_direc_all.append(kappa_J_J_avg)
                kappa_Jv_Jv_direc_all.append(kappa_Jv_Jv_avg)
                kappa_Jk_Jk_direc_all.append(kappa_Jk_Jk_avg)
                kappa_Jv_Jk_direc_all.append(kappa_Jv_Jk_avg)
                kappa_J_v_direc_all.append(kappa_J_v_avg)
                kappa_v_v_direc_all.append(kappa_v_v_avg)
                kappa_total_direc_all.append(kappa_total)
                #
                kappa_J_J_direc_all_error.append(kappa_J_J_avg_error)
                kappa_Jv_Jv_direc_all_error.append(kappa_Jv_Jv_avg_error)
                kappa_Jk_Jk_direc_all_error.append(kappa_Jk_Jk_avg_error)
                kappa_Jv_Jk_direc_all_error.append(kappa_Jv_Jk_avg_error)
                kappa_J_v_direc_all_error.append(kappa_J_v_avg_error)
                kappa_v_v_direc_all_error.append(kappa_v_v_avg_error)
                kappa_total_direc_all_error.append(kappa_error)

            kappa_J_J_mean = np.mean(kappa_J_J_direc_all, axis=0)
            kappa_Jv_Jv_mean = np.mean(kappa_Jv_Jv_direc_all, axis=0)
            kappa_Jk_Jk_mean = np.mean(kappa_Jk_Jk_direc_all, axis=0)
            kappa_Jv_Jk_mean = np.mean(kappa_Jv_Jk_direc_all, axis=0)
            kappa_J_v_mean = np.mean(kappa_J_v_direc_all, axis=0)
            kappa_v_v_mean = np.mean(kappa_v_v_direc_all, axis=0)
            kappa_total_mean = np.mean(kappa_total_direc_all, axis=0)

            kappa_J_J_mean_error = np.sqrt(np.sum(np.array(kappa_J_J_direc_all_error)**2))/3
            kappa_Jv_Jv_mean_error = np.sqrt(np.sum(np.array(kappa_Jv_Jv_direc_all_error)**2))/3
            kappa_Jk_Jk_mean_error = np.sqrt(np.sum(np.array(kappa_Jk_Jk_direc_all_error)**2))/3
            kappa_Jv_Jk_mean_error = np.sqrt(np.sum(np.array(kappa_Jv_Jk_direc_all_error)**2))/3
            kappa_J_v_mean_error = np.sqrt(np.sum(np.array(kappa_J_v_direc_all_error)**2))/3
            kappa_v_v_mean_error = np.sqrt(np.sum(np.array(kappa_v_v_direc_all_error)**2))/3
            kappa_total_mean_error = np.sqrt(np.sum(np.array(kappa_total_direc_all_error)**2))/3

            # save the main results
            file.write(f"{temperature}," f"{kappa_J_J_mean:.6e}±{kappa_J_J_mean_error:.6e}, "
                       f"{kappa_Jv_Jv_mean:.6e}±{kappa_Jv_Jv_mean_error:.6e}, "
                       f"{kappa_Jk_Jk_mean:.6e}±{kappa_Jk_Jk_mean_error:.6e}, "
                       f"{kappa_Jv_Jk_mean:.6e}±{kappa_Jv_Jk_mean_error:.6e}, "
                       f"{kappa_J_v_mean:.6e}±{kappa_J_v_mean_error:.6e}, "
                       f"{kappa_v_v_mean:.6e}±{kappa_v_v_mean_error:.6e}, "
                       f"{kappa_total_mean:.6e}±{kappa_total_mean_error:.6e}\n")


            print("Complete!")
