#!/bin/bash

usage()
{
 echo "howtoexecute.sh: wrong number of input parameters. Exiting."
 echo -e "Usage: bash howtoexecute.sh <supercomputer>"
 echo -e "  g.e: bash howtoexecute.sh ogbon"
}

nowherman()
{
 nvcc -I/usr/include/openmpi -L/usr/lib/openmpi -lmpi -Xcompiler -fopenmp ping-pong-CUDA-AWARE-MPI.cu -o ping-pong-CUDA-AWARE-MPI
}


ogbon()
{
 nvcc -I/opt/share/openmpi/4.1.0-cuda/include -L/opt/share/openmpi/4.1.0-cuda/lib64 -DprintLabel -lnccl -lmpi -Xcompiler -fopenmp ping-pong-CUDA-AWARE-MPI.cu -o ping-pong-CUDA-AWARE-MPI
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