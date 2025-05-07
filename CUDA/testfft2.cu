/*William Dellinger, started Thursday April 10 2025
Program aimed towards amping up the speed at which the data from our ADCs can be 
converted into LACSPC and LCCSPC correlation data*/
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <cufft.h>
#include <cuda_runtime.h>
//#include <cstdint.h>
#include <sys/stat.h> 
#include <assert.h>
#include <vector>
#include "goatedFilereader.h"
using namespace std;


/*function to read the input files , place them in a buffer vector , call the inline function on this vector 
and then place the result into a secondary vector which is then returned
*/

vector<packet> readData(char * filename){
    vector<packet> data = readFile(filename);
    return data;
}

//this function should solely handle calculation of the fft 
std::vector<cufftComplex> run_fft(vector<packet> data, int size) {
    int N = size;
    cufftHandle plan;
    cufftComplex *d_data;
    cudaMalloc(&d_data, sizeof(cufftComplex) * N);

    vector<cufftComplex> host_complex(N);


    // for (int i = 0; i < N; ++i) {
    //     host_complex[i].x = (float)host_data[i];
    //     host_complex[i].y = 0.0f;
    // }

    // for(auto i: data){
    //     for(int j = 0; j < 1024; j++){
    //         host_complex[i].
    //     }
    // }

    cudaMemcpy(d_data, host_complex.data(), sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);

    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    cufftExecR2C(plan, d_data, d_data, CUFFT_FORWARD);

    cudaDeviceSynchronize();

    vector<cufftComplex> result(N);
    cudaMemcpy(result.data(), d_data, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);
    
    cufftDestroy(plan);
    cudaFree(d_data);

    return result;
   
}


void writeOut(char * filename, int16_t ** data ){
    

}


//need the parameters in the main function, additonally i need all of the gpu dma stuff to occur in teh main function
int main (int argc, char * argv[]){

    packet* data = readData(argv[1]);
    run_fft(data, )



}


/*main() {
    // 1. Allocate memory
    // 2. Copy data to device
    runFFT(d_data, size);
    // 3. Optional postprocess kernel
    scaleFFT<<<blocks, threads>>>(d_data, size);
    // 4. Copy results back to host
}*/
