#include <iostream>
#include <fstream>
#include <cufft.h>
#include <cuda_runtime.h>
#include <struct.h>
#include <sys/stat.h> 



using namespace std;


//plan is to read the files, put them in vectors and then copy them to DEVICE then the device will perform the fft operation
//on all of the data 
inline int16_t convertToInt16(uint8_t * Bytes, int32_t offset){
    assert((offset & 1) == 0);
    return *(int16_t) &Bytes[offset];
};

vector<int16_t> getData(char *filename){
    vector<int16_t> data;
    char * dataBlock; 
    ifstream file(filename);
    if(file.is_open()){
        struct stat info;
        stat(filename, &info);
        const size_t size = info.st_size; 
        vector<char>buffer(size); 
         while (file.read(buffer.data(), size)){
         cout.write(buffer.data(), file.gcount())
         }

        file.close();
        vector<uint8_t> buffer2;
        for(int i = 0; i < size; i++){
            buffer2[i].push_back((uint8_t) buffer[i]);
        }

        for(int j = 0; j<size; j++){
            data.push_back(convertToInt16(buffer2[i], 0)); 
        }

        return data; 
        
  //read in the whole file using fseek and then use the inline function to convert the block to int16 and then read it into the vector to return :)      
         
}



__global__ void fft(vector<int16_t, int16_t> data)
{

 vector<int16_t, int16_t> a ** data;   
 cudaMalloc(&a, sizeof(data));
 cudaMemcpy(a, &data, sizeof(data), cudaMemcpyHostToDevice); 
 
 
 

}










int main (int argc, char * argv[]){


vector<int16_t, int16_t> data;

for(int i = 1; i< argc; i++){    
data[i] = getData(argv[i]);
}

fft<<<1,1>>>(data); 




}