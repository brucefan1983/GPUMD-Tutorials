#!/bin/bash

rm job*
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


conda activate Tools

ulimit -s unlimited

cd ${DIR}

python GK.py

END_OF_CAT

chmod 755 sub.out
sbatch < sub.out

