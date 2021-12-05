#!/bin/bash

#SBATCH --job-name=cuda-ware-mpi-laplacian  # Job name
#SBATCH --nodes=1                           # Run all processes on a single node   
#SBATCH --partition=GPUlongB                # partition OGBON
#SBATCH --output=out_%j.log                 # Standard output and error log

mpirun -np 2 -x UCX_MEMTYPE_CACHE=n  -mca pml ucx -mca btl ^vader,tcp,openib,smcuda -x UCX_NET_DEVICES=mlx5_0:1  ./cuda-aware-mpi-laplacian
