#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <string.h>


int16_t *  get_data(char * filename){
	struct stat filestat; //obtain struct of file info
	int16_t buffer;  
	stat(filename, &filestat); //function to get infor
	int16_t * data = malloc(sizeof(int16_t) * filestat.st_size/2); 
	FILE * fp = fopen(filename, "rb+");
	for(int i = 0; i< (filestat.st_size)/2; i++){
		fscanf(fp,"%hd", &buffer);
		data[i] = buffer; 
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



int16_t * DoFilter(int16_t * data){
		
		int size = sizeof(data)/2;
		int16_t * filteredData = malloc(sizeof(int16_t) * size); 
		for(int i = 0; i<size; i++){
			filteredData[i] = median(data[i-1], data[i], data[i+1]);
		}
		return filteredData; 
}



void writeData(int16_t * Filtered_Data, char * outFilename){
	FILE * fp = fopen(outFilename, "wb");
	int size = sizeof(Filtered_Data) / 2; 
	int16_t buffer;
	for(int i = 0; i < size; i++){
		buffer = Filtered_Data[i];
		fprintf(fp, "%hd", &buffer);
	}
	fclose(fp);
}


int main(int argc, char * argv[]){
	if(argc!=3){
		printf("Error Usage: this function only requires the name of the input file as the first parameter and the desired output file as the second. ");
		return -1;
	}
	char * filename = malloc(strlen(argv[1]) + 1);

	strcpy(filename,argv[1]);

	int16_t * data = get_data(filename);
	int16_t * Filtered_Data = DoFilter(data);

	free(Filtered_Data);
	free(data);
	free(filename); 

	return 0;
}











