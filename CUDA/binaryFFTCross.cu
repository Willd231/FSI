#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <cuda_runtime.h>

#define BYTES_PER_SAMPLE 2
#define BITS_PER_NUM 5
#define TWOCOMP_MAXUINT   (1<<(BITS_PER_NUM))    
#define TWOCOMP_ZEROPOINT (1<<(BITS_PER_NUM-1))  
#define DECODER_MASK ((1<<(2*BITS_PER_NUM)) -1)

__global__ void compute_CMAC(const int nchan, const int ninp, const int prod_type, cufftComplex* ft_buf, cufftComplex* corr_buf) {
    int inp1 = blockIdx.x;
    int inp2 = threadIdx.x;
    if (inp1 < ninp && inp2 < ninp) {
        for (int chan = 0; chan < nchan; chan++) {
            if (prod_type == 'B' || ((prod_type == 'C' && inp1 != inp2) || (prod_type == 'A' && inp1 == inp2))) {
                cufftComplex a = ft_buf[inp1 * nchan + chan];
                cufftComplex b = ft_buf[inp2 * nchan + chan];
                cufftComplex result;
                result.x = a.x * b.x + a.y * b.y; // Real part
                result.y = a.x * b.y - a.y * b.x; // Imaginary part
                corr_buf[inp1 * ninp * nchan + inp2 * nchan + chan] = result;
            }
        }
    }
}

void do_FFT_CUDA(int nchan, int ninp, cufftComplex* d_inp_buf, cufftComplex* d_ft_buf, cufftComplex* d_corr_buf, int prod_type) {
    cufftHandle plan;
    cufftPlan1d(&plan, nchan, CUFFT_C2C, ninp);
    cufftExecC2C(plan, d_inp_buf, d_ft_buf, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    dim3 blocks(ninp);
    dim3 threads(ninp);
    compute_CMAC<<<blocks, threads>>>(nchan, ninp, prod_type, d_ft_buf, d_corr_buf);
    cudaDeviceSynchronize();

    cufftDestroy(plan);
}

int readData(FILE* fpin, int nchan, int ninp, cufftComplex* inp_buf, int nskip) {
    int inp, chan, offset = 32;  // 32 int16 values to skip
    int16_t* sample = (int16_t*)malloc(sizeof(int16_t) * ninp * nchan);

    fseek(fpin, nskip * ninp * nchan * sizeof(int16_t), SEEK_CUR);

    if (fread(sample, sizeof(int16_t), ninp * nchan, fpin) != ninp * nchan) {
        fprintf(stderr, "Error reading data from file\n");
        return 1;  // Or any other error-handling mechanism
    }

    // Convert to float
    for (inp = 0; inp < ninp; inp++) {
        for (chan = 0; chan < nchan; chan++) {
            inp_buf[inp * nchan + chan].x = (float)(sample[inp * nchan + chan]) / 16.0f;  // Real part, scaled by 16
            inp_buf[inp * nchan + chan].y = 0.0f;  // Imaginary part set to 0 initially
        }
    }

    free(sample);
    return 0;
}

void writeOutput(cufftComplex* corr_buf, int ninp, int nchan, int prod_type, FILE* fout_ac, FILE* fout_cc, int iter) {
    int inp1, inp2, chan;
    float* temp_buffer = (float*)malloc(nchan * sizeof(float));
    float normaliser = 1.0 / (nchan * iter);  // Apply normalization factor

    for (inp1 = 0; inp1 < ninp; inp2++) {
        for (inp2 = inp1; inp2 < ninp; inp2++) {
            for (chan = 0; chan < nchan; chan++) {
                cufftComplex val = corr_buf[inp1 * ninp * nchan + inp2 * nchan + chan];
                val.x *= normaliser;
                val.y *= normaliser;

                if (inp1 == inp2 && (prod_type == 'A' || prod_type == 'B')) {
                    // Auto-correlation: only the real part
                    temp_buffer[chan] = val.x;
                }
            }

            if (inp1 == inp2 && (prod_type == 'A' || prod_type == 'B')) {
                // Write auto-correlation to .LACSPC
                fwrite(temp_buffer, sizeof(float), nchan, fout_ac);
            }
            if (inp1 != inp2 && (prod_type == 'C' || prod_type == 'B')) {
                // Write cross-correlation to .LCCSPC
                fwrite(&corr_buf[inp1 * ninp * nchan + inp2 * nchan], sizeof(cufftComplex), nchan, fout_cc);
            }

            // Reset correlation buffer to zero
            memset(&corr_buf[inp1 * ninp * nchan + inp2 * nchan], 0, nchan * sizeof(cufftComplex));
        }
    }

    free(temp_buffer);
}

int main(int argc, char* const argv[]) {
    int nchan = 1024;
    int ninp = 4;
    int nskip = 0;
    int prod_type = 'B';
    int iter = 99999;  // Iteration for averaging
    FILE *fpin = NULL, *fout_ac = NULL, *fout_cc = NULL;

    char lacspc_filename[FILENAME_MAX], lccspc_filename[FILENAME_MAX];

    // Command line parsing
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0) nchan = atoi(argv[++i]);
        if (strcmp(argv[i], "-n") == 0) ninp = atoi(argv[++i]);
        if (strcmp(argv[i], "-s") == 0) nskip = atoi(argv[++i]);
        if (strcmp(argv[i], "-p") == 0) prod_type = argv[++i][0];
        if (strcmp(argv[i], "-i") == 0) fpin = fopen(argv[++i], "rb");
        if (strcmp(argv[i], "-o") == 0) {
            strcpy(lacspc_filename, argv[++i]);
            strcpy(lccspc_filename, lacspc_filename);
            strcat(lacspc_filename, ".LACSPC");
            strcat(lccspc_filename, ".LCCSPC");
        }
    }

    if (fpin == NULL) {
        printf("No input file specified.\n");
        return -1;
    }

    // Open output files
    fout_ac = fopen(lacspc_filename, "wb");
    fout_cc = fopen(lccspc_filename, "wb");
    if (fout_ac == NULL || fout_cc == NULL) {
        printf("Error opening output files.\n");
        return -1;
    }

    // Allocate memory for input and correlation results
    cufftComplex* h_inp_buf = (cufftComplex*)malloc(ninp * nchan * sizeof(cufftComplex));
    cufftComplex* h_corr_buf = (cufftComplex*)malloc(ninp * ninp * nchan * sizeof(cufftComplex));

    // Read data from file and convert to float
    readData(fpin, nchan, ninp, h_inp_buf, nskip);
    fclose(fpin);

    // Allocate memory on the device
    cufftComplex *d_inp_buf, *d_ft_buf, *d_corr_buf;
    cudaMalloc((void**)&d_inp_buf, ninp * nchan * sizeof(cufftComplex));
    cudaMalloc((void**)&d_ft_buf, ninp * nchan * sizeof(cufftComplex));
    cudaMalloc((void**)&d_corr_buf, ninp * ninp * nchan * sizeof(cufftComplex));

    // Copy input data to device
    cudaMemcpy(d_inp_buf, h_inp_buf, ninp * nchan * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // Perform FFT and compute correlation products on the GPU
    do_FFT_CUDA(nchan, ninp, d_inp_buf, d_ft_buf, d_corr_buf, prod_type);

    // Copy correlation results back to host
    cudaMemcpy(h_corr_buf, d_corr_buf, ninp * ninp * nchan * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Write the results to output files
    writeOutput(h_corr_buf, ninp, nchan, prod_type, fout_ac, fout_cc, iter);

    // Free memory
    cudaFree(d_inp_buf);
    cudaFree(d_ft_buf);
    cudaFree(d_corr_buf);
    free(h_inp_buf);
    free(h_corr_buf);

    // Close output files
    fclose(fout_ac);
    fclose(fout_cc);

    return 0;
}




