
#include <iostream>
#include <cuda.h>

using namespace std;

__global__ void AddIntsCUDA(int * a, int * b)
{
a[0] += b[0];
}


int main (){

int a = 5, b = 9;

int *d_a, *d_b;

//then space for the data must be allocated on the gpu 

cudaMalloc(&d_a, sizeof(int));
cudaMalloc(&d_b, sizeof(int));

//and then go ahead and copy the actual data to the DEVICE once the memory is allocated

cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice); 

// and then call the kernel to actually perform the calculation once you have the memory allocated for the data and it copied over

AddIntsCUDA<<<1,1>>> (d_a, d_b); 


//following the calculation you must then copy the data back to the HOST 

cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

cout <<"The answer is "<<a<<endl;

return 0; 
} 
