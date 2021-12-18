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

 #SBATCH --job-name=ping-pong-MPI         # Job name
 #SBATCH --nodes=2                        # Run all processes on 2 nodes   
 #SBATCH --partition=GPUlongB             # partition OGBON
 #SBATCH --output=out_%j.log              # Standard output and error log

 mpirun -np 2 -x UCX_MEMTYPE_CACHE=n  -mca pml ucx -mca btl ^vader,tcp,openib,smcuda -x UCX_NET_DEVICES=mlx5_0:1  ./ping-pong-MPI

}


#args in comand line
if [ "$#" ==  1 ]; then
 usage
 exit
fi

#airis
if [[ $1 == "nowherman" ]];then
 nowherman
fi


#ogbon
if [[ $1 == "ogbon" ]];then
 ogbon
fi