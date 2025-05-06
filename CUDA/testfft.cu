/*William Dellinger, started Thursday April 10 2025
Program aimed towards amping up the speed at which the data from our ADCs can be 
converted into LACSPC and LCCSPC correlation data*/

#include <iostream>
#include <fstream>
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <sys/stat.h> 
#include <assert.h>
#include <vector>
using namespace std;



typedef struct adcData
{ 

int16_t * adc1;
int16_t * adc2;
int16_t * adc3;
int16_t * adc4;
}


//inline function converting from uint8 to int16 
inline int16_t convertToInt16(uint8_t * Bytes, int32_t offset){
    assert((offset & 1) == 0);
    return (int16_t) &Bytes[offset];
};


/*function to read the input files , place them in a buffer vector , call the inline function on this vector 
and then place the result into a secondary vector which is then returned
*/
int16_t * getData(char *filename){

    ifstream file(filename);
    if(file.is_open()){
        struct stat info;
        stat(filename, &info);
        const size_t size = info.st_size;
	int numpackets = size / 8256;
for(int i = 0; i < numpackets; i++){


}
        return data;

}
}

//this function should solely handle calculation of the fft 
void run_fft(int16_t * data, int size) {
    int N = host_data.size();
    cufftHandle plan;
    cufftComplex *d_data;
    cudaMalloc(&d_data, sizeof(cufftComplex) * N);

    vector<cufftComplex> host_complex(N);
    for (int i = 0; i < N; ++i) {
        host_complex[i].x = (float)host_data[i];
        host_complex[i].y = 0.0f;
    }

    cudaMemcpy(d_data, host_complex.data(), sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);

    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    cudaDeviceSynchronize();

    vector<cufftComplex> result(N);
    cudaMemcpy(result.data(), d_data, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);
    
    cufftDestroy(plan);
    cudaFree(d_data);

   
}


void writeOut(char * filename, int16_t ** data ){
    

}


//need the parameters in the main function, additonally i need all of the gpu dma stuff to occur in teh main function
int main (int argc, char * argv[]){

int nchan=128;      
int ninp=4;         

char *infilename=argv[1],*outfilename=argv[2];


int16_t * data = malloc(sizeof(int16_t) ); 


data = getData(argv[1]);



run_fft(data, size); 




}


/*main() {
    // 1. Allocate memory
    // 2. Copy data to device
    runFFT(d_data, size);
    // 3. Optional postprocess kernel
    scaleFFT<<<blocks, threads>>>(d_data, size);
    // 4. Copy results back to host
}*/
