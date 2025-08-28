#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/stat.h>


int16_t *  get_data(char * filename){
	stat(filename); 
	int16_t * data = malloc(sizeof(int16_t) * st_size); 
	FILE * fp = fopen(filename, "rb+");
	while(fp != NULL){
	
}



int16_t median(int16_t A, int16_t B, int16_t C){
		int16_t Median = NULL;
		if(A == B == C){ Median = A;}
		if(A > B == C){Median = B;}
		if(A < B == C){Median = B;}
		if(A == B > C){Median = A;}
		if(A == B < C){Median = A;}
		if(A > B < C){Median = C;}
		if(A < B > C){Median = A;}
		if(A<B<C){Median = B;}
		if(A>B>C){Median = B;}

		return Median;
	
}


int16_t * DoFilter(int16_t * data){
		
		int size = sizeof(data)/2;
		int16_t * filteredData = malloc(sizeof(int16_t) * size); 
		for(int i = 0; i<size; i++){
			filteredData[i] = median(i, i+1, i + 2);
		}
		return filteredData; 
}



void writeData(int16_t * Filtered_Data, char * outFilename){
	FILE * fp = fopen(outFilename, "wb+");
	int size = sizeof(Filtered_Data) / 2; 
	for(int i = 0; i < size; i++){
		fp.write(Filtered_Data[i]);
	}
}


int main(int argc, char * argv[]){
	if(argc <= 2 || argc >= 3){
		printf("Error Usage: this function only requires the name of the input file as the first parameter and the desired output file as the second. ");
	}
	char * filename = malloc(sizeof(char) * strlen(argv[1]));

	strpcy(argv[1], filename);

	int16_t * data = get_data(filename);
	int16_t * Filtered_Data = DoFilter(data);
	


	return 0;
}











