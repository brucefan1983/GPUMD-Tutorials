#!/bin/bash

rm job*
rm CHG* CONT* DOSCAR EIGENVAL IBZKPT ICONST OSZICAR OUTCAR PCDAT REPORT vasprun.xml WAVECAR XDATCAR
#--------------
PAR='main'
NOD=1
CPU_CORE=1
TIM='160:00:00'
#--------------
JOB=$1
DIR=`pwd`
#--------------

cat << END_OF_CAT > sub.out
#!/bin/bash

#SBATCH -o job.%j.out
#SBATCH --job-name=${JOB}
#SBATCH -p ${PAR}
#SBATCH --nodes=${NOD}
#SBATCH --ntasks-per-node=${CPU_CORE}
#SBATCH --gres=gpu:1                # 请求 1 个 GPU

module load oneapi/2023.0.0
export I_MPI_FABRICS=shm:ofi
module load GPUMD/202412
ulimit -s unlimited

cd ${DIR}

gpumd < run.in > run.out 

END_OF_CAT

chmod 755 sub.out
sbatch < sub.out


