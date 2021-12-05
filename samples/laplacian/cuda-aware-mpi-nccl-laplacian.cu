/*%****************************************************************************80
%  Code: 
%   cuda-aware-mpi-laplacian.cu
%
%  Purpose:
%   Implements sample 2D Laplacian Method in CUDA-AWARE MPI
%
%  Modified:
%   Dec 02 2021 10:57 
%
%  Author:
%    Murilo Boratto <murilo.boratto 'at' fieb.org.br>
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
#include <nccl.h>

__global__ void kernel(double *a,  double *c, double *stencil, int m, int n, int jsta, int jend) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;     
  int j = blockIdx.y * blockDim.y + threadIdx.y;    

  double sx, sz;
  double dx = 1, dz = 1;
     
  if(j >= jsta - 1 && j < jend && i >= 1 && i < (m - 1)){
      sx = a[(i-1) + j*n]  + a[(i+1)+ j*n]        + 2 * a[i + j*n];   
      sz = a[ i + (j-1)*n] + stencil[i + (j+1)*n] + 2 * a[i + j*n]; 
      c[i + j * n] = (sx/(dx*dx)) + (sz/(dz*dz));
  }

}/*kernel*/


void showMatrix(double *a, int n){

   for(int i = 0; i < n; i++){
      for(int j = 0; j < n; j++){
        printf("%1.2f\t", a[i + j*n]);
      }
    printf("\n");
   }

   printf("\n");

}/*showMatrix*/


void PARA_RANGE(int n1,int n2, int nprocs, int myid, int *vector_return){

	int iwork1 = (n2 - n1 + 1) / nprocs;
	int iwork2 = (n2 - n1 + 1) % nprocs;

	int jsta   = (myid * iwork1) + n1 + fmin((double) myid, (double) iwork2);
	int jend   = jsta + iwork1 - 1;

	if (iwork2 > myid)
	 jend = jend + 1;

  vector_return[0] = jsta;
  vector_return[1] = jend;

}/*PARA_RANGE*/


void SUPER_PARA_RANGE(int n, int nprocs, int myid, int *vector_return, double *host_a){   

  PARA_RANGE(1, n, nprocs, myid, vector_return);

	int jsta = vector_return[0];
	int jend = vector_return[1];
        
   for(int i = 0; i < n; i++)
	   for(int j = jsta-1; j < jend; j++) 
	     host_a[i + j* n] = (i + j + 2) * 1.;            
          
   //showMatrix(host_a, n); 

}/*SUPER_PARA_RANGE*/

void definitionParticionSendRecv(int *partition, int n, int nGPUs){

 int *vector_return = (int *) calloc (2, sizeof(int));
 int i = 0;

 for(int myid=0; myid < nGPUs; ++myid){
   PARA_RANGE(1, n, nGPUs, myid, vector_return);
   partition[i++] = vector_return[1];
 }

 free(vector_return);   

}/*definitionParticionNCCLSendRecv*/

void freeMemoryApp(int *DeviceList, cudaStream_t *s, ncclComm_t *comms, double **device_a, double **device_c, 
                   double **Solution_reduced_device, double **stencil, int nGPUs){

      for(int g = 0; g < nGPUs; g++){ 
         cudaSetDevice(DeviceList[g]);
         cudaStreamSynchronize(s[g]);
        }
    
        for(int g = 0; g < nGPUs; g++){  
         cudaSetDevice(DeviceList[g]);
         cudaStreamDestroy(s[g]);
        }

        for(int g = 0; g < nGPUs; g++)    
          ncclCommDestroy(comms[g]);
      
        cudaFree(device_a);
        cudaFree(device_c);
        cudaFree(Solution_reduced_device);
        cudaFree(stencil);

}/*freeMemoryApp*/


/******************************************** MAIN ***********************************************************************************/

int main(int argc, char *argv[]){

      int n = 8;                                  /*Size problem*/
      int nGPUs = 0;                              /*Initialize the GPU variable*/
      int rank, size;             
      cudaGetDeviceCount(&nGPUs);                /*Definitions the numbers of GPUs of the System*/
      int *DeviceList = (int *) malloc ( nGPUs * sizeof(int));
    
      /*MPI Initialization*/
      MPI_Init(&argc, &argv);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Status status; 

      for(int i = 0; i < nGPUs; ++i)             /*Adding number of GPUs in a list*/
        DeviceList[i] = i;

      /*General Variables*/
      double **device_a                 = (double**) malloc (nGPUs     * sizeof(double*));
      double **device_c                 = (double**) malloc (nGPUs     * sizeof(double*));
      double **stencil                  = (double**) malloc (nGPUs     * sizeof(double*));
      double **Solution_reduced_device  = (double**) malloc (nGPUs     * sizeof(double*));
       
      /*Inicializing NCCL*/
      ncclComm_t* comms = (ncclComm_t*)   malloc(nGPUs * sizeof(ncclComm_t));  
      cudaStream_t* s   = (cudaStream_t*) malloc(nGPUs * sizeof(cudaStream_t));
      ncclCommInitAll(comms, nGPUs, DeviceList);

      /* 2D GRID and SIZEBLOCK definitions*/
      int  sizeblock = n / 2 ;
      int  grid = (int) ceil( (double) n / (double) sizeblock );
      dim3 dimGrid( grid, grid );
      dim3 dimBlock(sizeblock, sizeblock);
   
/**************************************************************************************************************************/

/*Step 1 - Divison of Integral Domain of the Matrix A on GPUs*/

      for(int myid = 0; myid < nGPUs; ++myid){
     
        double  *host_a    =  (double*) calloc (n * n, sizeof(double)); 
        int     *vector_return =  (int *) calloc (2, sizeof(int));
    
        SUPER_PARA_RANGE(n, nGPUs, myid, vector_return, host_a);   

        cudaSetDevice(DeviceList[myid]);

        cudaMalloc(&device_a[myid],  n  * n * sizeof(double));  
	      cudaMalloc(&device_c[myid],  n  * n * sizeof(double));
	      cudaMalloc(&stencil[myid],   n  * n * sizeof(double));
        cudaMalloc(&Solution_reduced_device[myid],  n  * n * sizeof(double));   

        cudaMemcpy(device_a[myid], host_a,   n * n * sizeof(double), cudaMemcpyHostToDevice) ;
 
        free(host_a);
        free(vector_return);
     
       }

      #ifdef printLabel  
        if(rank ==0){
      		for(int myid = 0; myid < nGPUs; ++myid){
        		double  *host_PrintStep1  =   (double*) calloc (n * n, sizeof(double)); 
        		cudaMemcpy(host_PrintStep1, device_a[myid], n * n * sizeof(double), cudaMemcpyDeviceToHost);
        		printf("GPU=%d ************************************************************************************\n\n", myid);
        		showMatrix(host_PrintStep1, n); 
        		free(host_PrintStep1);
      		}
        }   
      #endif

/**************************************************************************************************************************/

/* Step 2 - Start the MPI_SEND - MPI_Recv */

    int *partition = (int *) calloc (nGPUs, sizeof(int));
    definitionParticionSendRecv(partition, n, nGPUs);

     // GPU 1 --> GPU 0
    cudaSetDevice(1);   
    MPI_Send(device_a[1] + partition[0]*n, n * n, MPI_DOUBLE, 0, 1000, MPI_COMM_WORLD); 
    cudaSetDevice(0);   
    MPI_Recv(stencil[0] + partition[0]*n, n * n, MPI_DOUBLE, 1, 1000, MPI_COMM_WORLD, &status);

     // GPU 2 --> GPU 1           
    cudaSetDevice(2);   
    MPI_Send(device_a[2] + partition[1]*n, n * n, MPI_DOUBLE, 1, 1000, MPI_COMM_WORLD); 
    cudaSetDevice(1);   
    MPI_Recv(stencil[1]  + partition[1]*n, n * n, MPI_DOUBLE, 2, 1000, MPI_COMM_WORLD, &status);
        
     // GPU 3 --> GPU 2
    cudaSetDevice(3);   
    MPI_Send(device_a[3]+ partition[2]*n, n * n, MPI_DOUBLE, 2, 1000, MPI_COMM_WORLD); 
    cudaSetDevice(2);   
    MPI_Recv(stencil[2] + partition[2]*n, n * n, MPI_DOUBLE, 3, 1000, MPI_COMM_WORLD, &status);

 #ifdef printLabel  
  if(rank == 0) { 
    	 for(int mystencil = 0; mystencil < nGPUs-1; ++mystencil){      
       		double  *host_PrintStep2  =  (double*) calloc (n * n, sizeof(double)); 
       		cudaMemcpy(host_PrintStep2, stencil[mystencil], n * n * sizeof(double), cudaMemcpyDeviceToHost);
       		printf("Stencil=%d ************************************************************************************\n\n", mystencil); 
       		showMatrix(host_PrintStep2, n); 
       		free(host_PrintStep2); 
      	 }
   }
 #endif

/***************************************************************************************************************************************/

/* Step 3 - Building kernel CUDA */

ncclGroupStart(); 

    for(int myid = 0; myid < nGPUs; ++myid){

        cudaSetDevice(DeviceList[myid]);
        cudaStreamCreate(&s[myid]);
        
        int *vector_return = (int *) calloc (2, sizeof(int));

        PARA_RANGE(1, n, nGPUs, myid, vector_return);

        int jsta = vector_return[0];
        int jend = vector_return[1] + 1;
 
        if(myid == 0)
          jsta = 2;

        if(myid == (nGPUs - 1))
          jend = n - 1;
       
       /***********************************************************************************************************/
       /**/  kernel<<< dimGrid, dimBlock >>>(device_a[myid], device_c[myid], stencil[myid], n, n, jsta, jend);  /**/
       /***********************************************************************************************************/
        
        cudaSetDevice(DeviceList[myid]);
           
        ncclReduce(device_c[myid], Solution_reduced_device[myid], n * n, ncclDouble, ncclSum, 0, comms[myid], s[myid]);
       
        free(vector_return);

    }

ncclGroupEnd(); 

#ifdef printLabel
    double  *host_PrintStep3  =  (double*) calloc(n * n, sizeof(double)); 
    cudaMemcpy(host_PrintStep3, Solution_reduced_device[0], n * n * sizeof(double), cudaMemcpyDeviceToHost);
    printf("LAPLACIAN ************************************************************************************\n\n");   
    showMatrix(host_PrintStep3, n); 
    free(host_PrintStep3);
#endif

/**************************************************************************************************************************/

 freeMemoryApp(DeviceList, s, comms, device_a, device_c, Solution_reduced_device, stencil, nGPUs);

 //MPI_Finalize();
 MPI_Abort()
 
 return 0;

}/*main*/
