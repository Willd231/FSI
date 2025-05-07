#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <stdint.h>
#include <stdlib.h>


float * getdat(char * filename, float nspec){
//a note for this one - remember that the first index should be blank for my idea so increment starting at 1 and also that 


FILE *  fp = fopen(filename, "rb+");
//keep in mind that the data is in the form of float 32 written to bytes - 4 byte timestamp 
//get the size of the file in order to allocate correcly

//allocate an array to store

int i = 0;
int32_t value;
float * arr = malloc(sizeof(float) * nspec);
while (fread(&value, sizeof(int32_t), 1,fp)==1){
float fvalue = (float) value;
arr[i] = fvalue;
i+=1;
}

return arr;

}




int * process(char * filename[], nspec){

    int * data[]= getdat(filename, nspec);
    int size = getdat(filename);

    for(int i = 1; i < size; i++ ){
        data[i] =  median(data[i - 1], data[i], data[i+1]);
    }
return(data);

}


//function to find the median of three numbers 
int median(int a, int b, int c){

    int median; 
    if(a == b && a == c){
        median = a; 
    }
    if(a < b && b < c || c < b && b < a){
        median = b;
    }
    if(b < a && a < c || c < a && a < b){
        median = a;
    }
    if(a < c && c < a || b < c && c < a){
        median = b;
    }
    return median; 
}







int main (int argc, char * argv[]){
char * filename = argv[1];
struct stat file_stat;
int ninp = 4;
int nchan = 512;
int timestamp_size = 4;

int data_block_size = nchan * ninp * 4 + 4 * nchan;
long file_size = file_stat.st_size;
float fs = (float)file_size;
float nspec = fs / (data_block_size + timestamp_size);
printf("%f", &nspec);




process(filename, nspec);

//then write to file

fopen("median-filtered-data.text", "rb+");




    return 0;
}



