#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <cuda_runtime.h>


__global__ void printing(char * message){
printf("%s", message);
}


int main ()
{
char * message = new char[100];
printf("Type the message:/n ");
scanf("%99s", message);


char * d_message; //gpu v
cudaMalloc((void**)&d_message, 100 * sizeof(char));

//then you have to copy it to the gpu 
cudaMemcpy(d_message, message, 100 * sizeof(char), cudaMemcpyHostToDevice);


//then you have to launch the kernel 
printing<<<1,1>>>(message);
//this ensures the gpu finishes processing 
cudaDeviceSynchronize();

//then finally free all of the allocated mema

cudaFree(d_message);

return 0;
}
