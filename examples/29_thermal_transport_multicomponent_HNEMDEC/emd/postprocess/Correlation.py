import numpy as np
import torch
import os
import sys
import time
from functools import wraps
from multiprocessing import Pool
K_B = 8.617343e-5
PRESSURE_UNIT_CONVERSION = 1.602177e+2  # from natural to GPa
TIME_UNIT_CONVERSION = 1.018051e+1     # from natural to fs
KAPPA_UNIT_CONVERSION = 1.573769e+5    # from natural to W/mK
L1q_UNIT_CONVERSION = 1631.0961499964144
L11_UNIT_CONVERSION = 16.905134572911963


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"function {func.__name__} cost {end - start:.3f}s")
        sys.stdout.flush()
        return result
    return wrapper


@time_it
def get_compute(dirname: str):
    compute_files = []
    computes = []
    for root, _, files in os.walk(dirname):
        if "compute.out" in files:
            compute_file = os.path.join(root, "compute.out")
            compute_files.append(compute_file)

    with Pool() as pool:
        computes = pool.map(np.loadtxt, compute_files)

    computes = np.stack(computes)
    return computes


def save(filename, time, ac, rtc):
    array_saved = np.concatenate([time.reshape(-1, 1), ac, rtc], axis=-1)
    np.savetxt(filename, array_saved, fmt="%25.15e")


class CORRELATION():
    def __init__(self,
                 temperature,
                 time_step,
                 volume,
                 sampling_interval,
                 correlation_steps,
                 output_interval,
                 runs,
                 device="cpu"):
        # set device
        self.device = device

        # md parameters
        self.temperature = temperature
        self.time_step = time_step / TIME_UNIT_CONVERSION
        self.volume = volume

        # sampling parameters
        self.sampling_interval = sampling_interval
        self.Nc = correlation_steps
        self.output_interval = output_interval
        self.runs = runs

        self.Nd = int(self.runs / self.sampling_interval)  # number of samples
        self.dt = self.time_step * self.sampling_interval
        self.dt_in_ps = self.dt * TIME_UNIT_CONVERSION / 1000.0  # ps
        self.correlation_time = (torch.arange(self.Nc, device=self.device) +
                                 0.5*self.output_interval) * self.dt_in_ps

        # convert to W/mK
        factor_hh = self.dt * 0.5 / (K_B * self.temperature * self.temperature *
                                     self.volume) * KAPPA_UNIT_CONVERSION

        # convert to 10e-6 kg/smK
        factor_hm = self.dt * 0.5 / (K_B * self.temperature * self.temperature *
                                     self.volume) * L1q_UNIT_CONVERSION

        # convert to 10e-12 kgs/m^3k
        factor_mm = self.dt * 0.5 / (K_B * self.temperature * self.temperature *
                                     self.volume) * L11_UNIT_CONVERSION
        self.factor = {"HH": factor_hh,
                       "HM": factor_hm,
                       "MH": factor_hm,
                       "MM": factor_mm}

    @time_it
    def compute_correlation(self, flux_left, flux_right, couple_type):
        """
        correlation(t) = <flux_left(t)*flux_right(0)>,
        flux should have shape of (self.Nd, ...),
        couple_type:
            "HH": heat-heat,
            "HM": heat-mass,
            "MM": mass-mass
        """
        assert flux_left.shape == flux_right.shape
        factor = self.factor[couple_type]
        correlation = torch.zeros(
            (self.Nc,)+flux_left.shape[1:], device=self.device)
        rtc = torch.zeros((self.Nc,)+flux_left.shape[1:], device=self.device)
        for t in range(self.Nc):
            number_of_data = self.Nd - t
            correlation[t] = (flux_left[:number_of_data,]
                              * flux_right[t:]).mean(dim=0)
        pre_compute = factor*(correlation[:-1]+correlation[1:])
        rtc[1:] = torch.cumsum(pre_compute, dim=0)
        return correlation.to("cpu").numpy(), rtc.to("cpu").numpy()


if __name__ == "__main__":
    # set device
    device = "cuda"

    # md parameters
    temperature = 115  # K
    time_step = 1  # fs
    volume = 13824  # A^3
    group_num = 2  # group==component

    # sampling parameters
    sampling_interval = 1
    correlation_steps = 3000
    output_interval = 1
    runs = 50000
    compute = get_compute("../md")
    compute = torch.from_numpy(compute).to(device)

    # read jp, jk, momentum (suppose to be isotropic)
    shape = (runs, -1, group_num)
    compute = np.swapaxes(compute, 1, 0)
    jp = compute[..., :group_num*3].reshape(shape)
    jk = compute[..., group_num*3:group_num*6].reshape(shape)
    momentum = compute[..., group_num*6:group_num*9].reshape(shape)

    # total flux
    jt = jp+jk

    # initilize correlation function
    correlation = CORRELATION(temperature, time_step, volume, sampling_interval,
                              correlation_steps, output_interval, runs, device)
    correlation_time = correlation.correlation_time
    correlation_time = correlation_time.to("cpu").numpy()

    HH = "HH"
    HM = "HM"
    MM = "MM"

    '''
    # calculate (jp,jk)
    jpjp, rtc_jpjp = correlation.compute_correlation(
        jp.sum(axis=-1), jp.sum(axis=-1), HH)
    jkjp, rtc_jkjp = correlation.compute_correlation(
        jk.sum(axis=-1), jp.sum(axis=-1), HH)
    jpjk, rtc_jpjk = correlation.compute_correlation(
        jp.sum(axis=-1), jk.sum(axis=-1), HH)
    jkjk, rtc_jkjk = correlation.compute_correlation(
        jk.sum(axis=-1), jk.sum(axis=-1), HH)

    # save
    save("cor_pp.out", correlation_time, jpjp, rtc_jpjp)
    save("cor_kp.out", correlation_time, jkjp, rtc_jkjp)
    save("cor_pk.out", correlation_time, jpjk, rtc_jpjk)
    save("cor_kk.out", correlation_time, jkjk, rtc_jkjk)

    # calculate (jp,m)
    jpjm, rtc_jpjm = correlation.compute_correlation(
        jp.sum(axis=-1), momentum[..., 0], HM)
    jmjm, rtc_jmjm = correlation.compute_correlation(
        momentum[..., 0],  momentum[..., 0], MM)
    jmjp, rtc_jmjp = correlation.compute_correlation(
        momentum[..., 0], jp.sum(axis=-1), HM)

    # save
    save("cor_pm.out", correlation_time, jpjm, rtc_jpjm)
    save("cor_mm.out", correlation_time, jmjm, rtc_jmjm)
    save("cor_mp.out", correlation_time, jmjp, rtc_jmjp)

    # calculate (jk,m)
    jkjm, rtc_jkjm = correlation.compute_correlation(
        jk.sum(axis=-1), momentum[..., 0], HM)
    jmjk, rtc_jmjk = correlation.compute_correlation(
        momentum[..., 0], jk.sum(axis=-1), HM)

    # save
    save("cor_km.out", correlation_time, jkjm, rtc_jkjm)
    save("cor_mk.out", correlation_time, jmjk, rtc_jmjk)
    '''

    # calculate (jt,m)
    jtjt, rtc_jtjt = correlation.compute_correlation(
        jt.sum(axis=-1), jt.sum(axis=-1), HH)
    jtjm, rtc_jtjm = correlation.compute_correlation(
        jt.sum(axis=-1), momentum[..., 0], HM)
    jmjt, rtc_jmjt = correlation.compute_correlation(
        momentum[..., 0], jt.sum(axis=-1), HM)
    jmjm, rtc_jmjm = correlation.compute_correlation(
        momentum[..., 0],  momentum[..., 0], MM)

    # save
    save("cor_tt.out", correlation_time, jtjt, rtc_jtjt)
    save("cor_tm.out", correlation_time, jtjm, rtc_jtjm)
    save("cor_mt.out", correlation_time, jmjt, rtc_jmjt)
    save("cor_mm.out", correlation_time, jmjm, rtc_jmjm)

