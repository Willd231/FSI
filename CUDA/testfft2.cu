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
//#include "goatedFilereader.h"
using namespace std;

typedef struct packet{
    std::vector<int16_t> adc1, adc2, adc3, adc4;
}packet;


/*function to read the input files , place them in a buffer vector , call the inline function on this vector 
and then place the result into a secondary vector which is then returned
*/

// vector<packet> readData(char * filename){
//     vector<packet> data = readFile(filename);
//     return data;
// }

vector<packet> readFile(char * filename)
{
    
    cout << "In readFile function\n";

    ifstream file(filename, ios::binary);

    // check if file opened successfully
    if(!file.is_open()){
        cerr << "Error: Cannot open file " << filename << endl;
        return {};  // return empty vector
    }

    cout << "File opened\n";
    
    file.seekg(0, ios::end);
    size_t filesize = file.tellg();
    file.seekg(0, ios::beg);

    // debug:
    cout << "File size: " << filesize << endl;

    // debug: check if file size is multiple 8256
    if(filesize % 8256 != 0){
        cerr << "Warning: File size not a multiple of 8256 bytes.\n";
    }
    //----------

    size_t numPackets = filesize / 8256;

    // debug
    cout << "Number of packets: " << numPackets << endl;

    vector<packet> packets(numPackets);

    // allocate memory for inputs for each packet
    for(size_t i = 0; i < numPackets; i++){
        packets[i].adc1.resize(1024);
        packets[i].adc2.resize(1024);
        packets[i].adc3.resize(1024);
        packets[i].adc4.resize(1024);

    }

    file.seekg(64, std::ios::beg);

    for(size_t i = 0; i < numPackets; i++){
        // for (int j = 0; j < 1024; j++){
        //     file.read(reinterpret_cast<char*>(&(packets[i].adc1[j])), sizeof(int16_t));
        //     file.read(reinterpret_cast<char*>(&(packets[i].adc2[j])), sizeof(int16_t));
        //     file.read(reinterpret_cast<char*>(&(packets[i].adc3[j])), sizeof(int16_t));
        //     file.read(reinterpret_cast<char*>(&(packets[i].adc4[j])), sizeof(int16_t));

        // }
        for (size_t j = 0; j < 1024; j++) {
            if (!file.read(reinterpret_cast<char*>(&packets[i].adc1[j]), sizeof(int16_t)) ||
                !file.read(reinterpret_cast<char*>(&packets[i].adc2[j]), sizeof(int16_t)) ||
                !file.read(reinterpret_cast<char*>(&packets[i].adc3[j]), sizeof(int16_t)) ||
                !file.read(reinterpret_cast<char*>(&packets[i].adc4[j]), sizeof(int16_t))) {
                cerr << "Error: Incomplete or corrupted file at packet " << i << ", sample " << j << endl;
                return packets;
            }
        }


    }

    return packets;
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

    //cufftExecR2C(plan, d_data, d_data, CUFFT_FORWARD);

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


    // compile: nvcc testfft2.cu -lcufft -o testfft2
    //test directory: /mnt/carsedat/20250220_181537_0000.dat
    // run: ./testfft2 /mnt/carsedat/20250220_181537_0000.dat

    vector<packet> data = readFile(argv[1]);
    cout << "Data read in\n";

}


/*main() {
    // 1. Allocate memory
    // 2. Copy data to device
    runFFT(d_data, size);
    // 3. Optional postprocess kernel
    scaleFFT<<<blocks, threads>>>(d_data, size);
    // 4. Copy results back to host
}*/
