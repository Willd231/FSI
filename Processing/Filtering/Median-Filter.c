#include <stdio.h>
#include <stdlib.h>
#include <struct.h>
#include <sys/stat.h>


typedef struct the-data{
int16 * data;
int size;
} the-data;



int16 * getData(char * filename);
int16 * doFilter(int16 * data); 
struct * init(char * filename);

int main(int argc, char * argv[]){

if (argc <= 1){
	printf("Error Usage: filename required);
 return;
}

the-data * datainfo = malloc(sizeof(the-data));


int size = stat(argv[1]);

datainfo -> size = size; 

int16 * data = malloc(sizeof(int16) * size);

data = getData(argv[1]);

int16 * 

processedData = doFilter(data);
}






int16 * getData(char * filename, struct datainfo){
	FILE * fp = fopen(fp, "rb");
	
	while(fp != //eof){
		fscanf(fp, "%lf", datainfo->data[i]);
	}
//return data 
}




int16 * doFilter(int16 * data){
	/scoot this along from index zero , grabbing three at a time and taking the med
	int16 * FOR  = malloc(sizeof(int16)* 3); 
	

}


int16 getMedian(int16 * FOR){
	int16 median = NULL;
			if(FOR[0] == FOR[1] == FOR[2]){
				median = FOR[0];
			}
			if(FOR[0] > FOR[1] == FOR[2]){
				median = FOR[1];
			}
			if(FOR[0] < FOR[1] == FOR[2]){
				median = FOR[1];	
			}
			if(FOR[0] == FOR[1] > FOR[2]){
				median = FOR[0];	
			}
			if(FOR[0] == FOR[1] < FOR[2]){
				median = FOR[0];
			}
			if(FOR[0] > FOR[1] < FOR[2]){
				median = FOR[2];
			}
			if(FOR[0] < FOR[1] > FOR[2]){
				median = FOR[0];
			}
			if(FOR[0] < FOR[1] < FOR[2]){
				median = FOR[1];
			}
			if(FOR[0] > FOR[1] > FOR[2]){
				median = FOR[1];
			}
	return median;
}

	


































