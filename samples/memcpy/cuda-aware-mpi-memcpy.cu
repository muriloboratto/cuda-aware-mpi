/*%****************************************************************************80
%  Code: 
%   cuda-aware-mpi-memcpy.cu
%
%  Purpose:
%   Implements sample code using cuda-aware-mpi-memcpy.
%   The code allocate memory with cudaMalloc on GPUs.
%
%  Modified:
%   Dec 02 2021 10:57 
%
%  Author:
%    Murilo Boratto <murilo.boratto 'at' fieb.org.br>
%    Silvano Júnior <silvano.junior 'at' fieb.org.br>:
%
%  How to Compile:
%   bash howtocompile.sh
%
%  Execute: 
%   bash howtorun.sh
%
%****************************************************************************80*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <unistd.h>
#include <mpi.h>

#define N 16777216

int main(int argc, char **argv){

    int rank,size;
    double start, time; 
    
    MPI_Status status;
    
    size_t size_buffer = sizeof(float) * N;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
    float *buffer_h  =  (float*) malloc (size_buffer);
    float *buffer_h1 =  (float*) malloc (size_buffer);
    float *buffer_h2 =  (float*) malloc (size_buffer);
    float *buffer_h3 =  (float*) malloc (size_buffer);

    float *buffer_d  =  (float*) malloc (size_buffer);
    float *buffer_d1 =  (float*) malloc (size_buffer);
    float *buffer_d2 =  (float*) malloc (size_buffer);
    float *buffer_d3 =  (float*) malloc (size_buffer);

    if(rank==0){

        cudaSetDevice(0);    
        cudaMalloc((void **) &buffer_d, size_buffer);

        cudaSetDevice(1);
        cudaMalloc((void **) &buffer_d1, size_buffer);

        cudaSetDevice(2);
        cudaMalloc((void **) &buffer_d2, size_buffer);

        cudaSetDevice(3);
        cudaMalloc((void **) &buffer_d3, size_buffer);

        for(int i=0; i<N; i++)
            buffer_h[i] = i; 
    
        cudaSetDevice(0);
        cudaMemcpy(buffer_d,buffer_h,size_buffer,cudaMemcpyHostToDevice);
       
    }
    
    start = MPI_Wtime();

    for(int i=0; i<1000; i++){
        if(rank == 0){
            for(int count = 1; count < size; count++){
                cudaSetDevice(0); /*Send CUDA Buffer*/
                MPI_Send(buffer_h, N ,MPI_FLOAT,count,1000,MPI_COMM_WORLD);   
            }
        }else{
            
            if(rank == 1){
                cudaSetDevice(rank);
                MPI_Recv(buffer_h1, N ,MPI_FLOAT,0,1000,MPI_COMM_WORLD, &status);
            }

             if(rank == 2){
                 cudaSetDevice(rank);
                 MPI_Recv(buffer_h2, N ,MPI_FLOAT,0,1000,MPI_COMM_WORLD, &status);
             }

             if(rank == 3){
                 cudaSetDevice(rank);
                 MPI_Recv(buffer_h3, N ,MPI_FLOAT,0,1000,MPI_COMM_WORLD, &status);
             }
        } 
    }

    MPI_Finalize();
    
    time = MPI_Wtime() - start; 

    cudaMemcpy(buffer_h, buffer_d, size_buffer, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer_h1, buffer_d1, size_buffer, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer_h2, buffer_d1, size_buffer, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer_h3, buffer_d1, size_buffer, cudaMemcpyDeviceToHost);
    
    printf("Time: %f - rank %d, buffer %f \n ", time, rank, buffer_h[N-1]);
    printf("Time: %f - rank %d, buffer %f \n ", time, rank, buffer_h1[N-1]);
    printf("Time: %f - rank %d, buffer %f \n ", time, rank, buffer_h2[N-1]);
    printf("Time: %f - rank %d, buffer %f \n ", time, rank, buffer_h3[N-1]);
    
    free(buffer_h); 
    free(buffer_h1); 
    free(buffer_h2); 
    free(buffer_h3); 
    
    cudaFree(buffer_d);
    cudaFree(buffer_d1);
    cudaFree(buffer_d2);
    cudaFree(buffer_d3);

    return 0;

}/*main*/
