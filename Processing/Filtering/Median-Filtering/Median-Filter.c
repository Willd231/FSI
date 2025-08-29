#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <string.h>


int16_t *  get_data(char * filename){
    struct stat filestat; //obtain struct of file info  
    stat(filename, &filestat); //function to get infor
    int16_t * data = malloc(sizeof(int16_t) * filestat.st_size/2); 
    FILE * fp = fopen(filename, "rb");  
    
    for(int i = 0; i< (filestat.st_size)/2; i++){
        fread(&data[i], 2, 1, fp); 
         
    }
    fclose(fp);
    return data;
}



int16_t median(int16_t a, int16_t b, int16_t c){
    if (a > b) { int16_t t = a; a = b; b = t; }
    if (b > c) { int16_t t = b; b = c; c = t; }
    if (a > b) { int16_t t = a; a = b; b = t; }
    return b;
}



int16_t * DoFilter(int16_t * data, int length){
      
   
    int16_t * filteredData = malloc(sizeof(int16_t) * length); 
    for(int i = 0; i<length; i++){
    if(i == 0 || i == length -1){
    	filteredData[i] = data[i];
    }
    else{
        filteredData[i] = median(data[i-1], data[i], data[i+1]);  
        }
    }
    return filteredData; 
}



void writeData(int16_t * Filtered_Data, char * outFilename, int length){
    FILE * fp = fopen(outFilename, "wb");  
    
    
    for(int i = 0; i < length; i++){

	fwrite(&Filtered_Data[i], 2 , 1, fp); 
}
            
    fclose(fp);
}


int main(int argc, char * argv[]){
    if(argc!=3){
        printf("Error Usage: this function only requires the name of the input file as the first parameter and the desired output file as the second. ");
        return -1;
    }


      struct stat filestat;
      stat(argv[1], &filestat);
    	int len_data = filestat.st_size / sizeof(int16_t);
   
    int16_t * data = get_data(argv[1]);
    int16_t * Filtered_Data = DoFilter(data, len_data);
	writeData(Filtered_Data, argv[2], len_data);
	
   

    return 0;
}
