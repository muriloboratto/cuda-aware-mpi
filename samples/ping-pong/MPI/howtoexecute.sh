#!/bin/bash

usage()
{
 echo "howtoexecute.sh: wrong number of input parameters. Exiting."
 echo -e "Usage: bash howtoexecute.sh <supercomputer>"
 echo -e "  g.e: bash howtoexecute.sh ogbon"
}


nowherman()
{
 mpirun -np 2 ./ping-pong-MPI
}

ogbon()
{

cat << EOF > slurm-MPI.sh

#SBATCH --job-name=CUDA-AWARE-MPI              # Job name
#SBATCH --nodes=2                              # Run all processes on 2 nodes  
#SBATCH --partition=CPUlongB                   # Partition OGBON
#SBATCH --output=out_%j.log                    # Standard output and error log
#SBATCH --ntasks-per-node=1                    # Define 1 job per node
#SBATCH --account=cenpes-lde                   # Account user's OGBON

module load openmpi/4.1.1-cuda

mpirun -np 2 -x UCX_MEMTYPE_CACHE=n  -mca pml ucx -mca btl ^vader,tcp,openib,smcuda -x UCX_NET_DEVICES=mlx5_0:1  ./ping-pong-CUDA-AWARE-MPI

EOF

sleep .1

sbatch slurm-MPI.sh

}


#args in comand line
if [ "$#" ==  0 ]; then
 usage
 exit
fi

#nowherman
if [[ $1 == "nowherman" ]];then
 nowherman
fi


#ogbon
if [[ $1 == "ogbon" ]];then
 ogbon
fi