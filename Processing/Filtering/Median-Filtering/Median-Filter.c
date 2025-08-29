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
    // ❌ Mistake: opened in "rb+" but you're not writing. Also using fscanf below.
    // 💡 Hint: if the file is binary, open with "rb" and use fread instead of fscanf.

    for(int i = 0; i< (filestat.st_size)/2; i++){
        fscanf(fp,"%hd", &buffer);  
        // ❌ Mistake: fscanf expects text numbers, but the file is binary.
        // 💡 Hint: use fread for raw binary data instead.
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
    // ❌ Mistake: sizeof(data) gives size of pointer (usually 8), not array length.
    // 💡 Hint: pass the array size into this function instead.

    int16_t * filteredData = malloc(sizeof(int16_t) * size); 
    for(int i = 0; i<size; i++){
        filteredData[i] = median(data[i-1], data[i], data[i+1]);  
        // ❌ Mistake: out-of-bounds at i=0 (data[-1]) and at last element (data[size]).
        // 💡 Hint: handle edges separately (use neighbors carefully).
    }
    return filteredData; 
}



void writeData(int16_t * Filtered_Data, char * outFilename){
    FILE * fp = fopen(outFilename, "wb");  
    int size = sizeof(Filtered_Data) / 2;  
    // ❌ Mistake: same issue, sizeof(Filtered_Data) gives pointer size, not array length.
    // 💡 Hint: pass the size into this function.

    int16_t buffer;
    for(int i = 0; i < size; i++){
        buffer = Filtered_Data[i];
        fprintf(fp, "%hd", &buffer);  
        // ❌ Mistake: using &buffer passes a pointer to fprintf, wrong type.
        // ❌ Mistake: fprintf writes text, but file should be binary.
        // 💡 Hint: use fwrite with &buffer instead.
    }
    fclose(fp);
}


int main(int argc, char * argv[]){
    if(argc!=3){
        printf("Error Usage: this function only requires the name of the input file as the first parameter and the desired output file as the second. ");
        return -1;
    }
    char * filename = malloc(strlen(argv[1]) + 1);  
    // ❌ Mistake: malloc not needed, you can just use argv[1] directly.
    // 💡 Hint: remove malloc and strcpy, just pass argv[1].

    strcpy(filename,argv[1]);

    int16_t * data = get_data(filename);
    int16_t * Filtered_Data = DoFilter(data);

    // ❌ Mistake: never wrote output data to file.
    // 💡 Hint: call writeData(Filtered_Data, argv[2], size).

    free(Filtered_Data);
    free(data);
    free(filename);  // ❌ Mistake: unnecessary since malloc wasn’t needed.
    // 💡 Hint: remove this too if you stop malloc'ing filename.

    return 0;
}
