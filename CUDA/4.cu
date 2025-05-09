#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <ctype.h>
#include <stdint.h>            // for int16_t
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#define MAX_THREADS        1
#define BYTES_PER_SAMPLE   2


// that i found in the docs 
#define CUDA_CHECK(call) do {                                 
    cudaError_t err = (call);                               
    if (err != cudaSuccess) {                               
        fprintf(stderr, "CUDA Error %s:%d: %s\n",        
                __FILE__, __LINE__, cudaGetErrorString(err)); 
        exit(EXIT_FAILURE);                                 
    }                                                      
} while(0)

#define CUFFT_CHECK(call) do {                              
    cufftResult res = (call);                               
    if (res != CUFFT_SUCCESS) {                             
        fprintf(stderr, "cuFFT Error %s:%d: %d\n",        
                __FILE__, __LINE__, res);                 
        exit(EXIT_FAILURE);                                 
    }                                                  
//took out the CMAC just to focus on the FFT

typedef struct {
    cufftHandle *plans;
    int          doneplan;
} cufftplan;

typedef struct {
    int            init;
    int            ntoread;
    unsigned char *buffer;
} datareader_t;

/* globals */
int  nchan        = 128;
int  ninp         = 4;
int  debug        = 0;
int  naver        = 99999;
int  prod_type    = 'B';
char *infilename  = NULL;
char *outfilename = NULL;
int  nskip        = 0;

/* prototypes */
void   print_usage(char *const argv[]);
void   parse_cmdline(int argc, char *const argv[], const char *optstring);
void   openFiles(const char *infile, const char *outfile, int prod_type,
                 FILE **fin, FILE **fout_ac, FILE **fout_cc);
float  elapsed_time(struct timeval *start);
int    readData(datareader_t *r, int nchan, int ninp,
                FILE *fpin, cufftComplex **inp_buf);
void   do_FFT_fftw(cufftplan *plan, int nchan, int ninp,
                   cufftComplex **inp_buf, cufftComplex **ft_buf);
void   writeOutput(FILE *fout_ac, FILE *fout_cc, int ninp,
                   int nchan, int naver, int prod_type,
                   cufftComplex **buf, float normaliser);

int main(int argc, char *const argv[]) {
    int             filedone = 0, res = 0, iter = 0, nav_written = 0;
    int             ncorr = ninp*(ninp+1)/2;
    FILE           *finp = NULL, *fout_ac = NULL, *fout_cc = NULL;
    float           normaliser = 1.0f;
    float           read_time = 0, fft_time = 0, write_time = 0;
    cufftComplex  **inp_buf = NULL, **ft_buf = NULL;
    struct timeval  tv;
    cufftplan       fftplan[MAX_THREADS];
    datareader_t    reader[MAX_THREADS];

    for (int t = 0; t < MAX_THREADS; t++) {
        fftplan[t].doneplan = 0;
        fftplan[t].plans    = NULL;
        reader[t].init      = 0;
        reader[t].buffer    = NULL;
    }

    if (argc < 2) print_usage(argv);
    parse_cmdline(argc, argv, "dc:i:o:n:a:p:s:");

    if (!infilename || !outfilename) {
        fprintf(stderr, "Error: must specify -i and -o\n");
        print_usage(argv);
    }

    if (debug) {
        fprintf(stderr,
                "ARGS: inputs=%d corrProds=%d chans=%d prodType=%c\n"
                "      infile=%s outfile=%s\n",
                ninp, ncorr, nchan, prod_type,
                infilename, outfilename);
    }

    openFiles(infilename, outfilename, prod_type,
              &finp, &fout_ac, &fout_cc);

    
    CUDA_CHECK(cudaFree(0));
    if (debug) fprintf(stderr, "[debug] CUDA runtime initialized\n");

    // allocate input & FFT buffers
    inp_buf = (cufftComplex**)calloc(ninp, sizeof(cufftComplex*));
    ft_buf  = (cufftComplex**)calloc(ninp, sizeof(cufftComplex*));
    if (!inp_buf || !ft_buf) { perror("calloc"); exit(EXIT_FAILURE); }

    for (int i = 0; i < ninp; i++) {
        CUDA_CHECK(cudaMallocManaged((void**)&inp_buf[i], nchan * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMallocManaged((void**)&ft_buf[i],  nchan * sizeof(cufftComplex)));
        // debug 
        if (!inp_buf[i] || !ft_buf[i]) {
            fprintf(stderr, "Error: Failed to allocate memory for buffers\n");
            exit(EXIT_FAILURE);
        }
    }

    // debug
    for (int i = 0; i < ninp; i++) {
        if (!inp_buf[i] || !ft_buf[i]) {
            fprintf(stderr, "Error: Null buffer detected for input %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // skip initial spectra if requested
    for (int i = 0; i < nskip; i++) {
        readData(&reader[0], nchan, ninp, finp, inp_buf);
    }

    // main processing loop
    while (!filedone) {
        gettimeofday(&tv, NULL);
        res = readData(&reader[0], nchan, ninp, finp, inp_buf);
        read_time += elapsed_time(&tv);
        if (res) filedone = 1;

        if (!filedone) {
            if (debug) fprintf(stderr, "[debug] about to do FFT\n");
            gettimeofday(&tv, NULL);
            do_FFT_fftw(&fftplan[0], nchan, ninp, inp_buf, ft_buf);
            CUDA_CHECK(cudaDeviceSynchronize());
            fft_time += elapsed_time(&tv);
        }

        if ((filedone && !nav_written) || ++iter == naver) {
            if (debug) fprintf(stderr, "[debug] about to writeOutput\n");
            gettimeofday(&tv, NULL);
            normaliser = 1.0f / (nchan * iter);
            writeOutput(fout_ac, fout_cc, ninp, nchan, iter,
                        prod_type, ft_buf, normaliser);
            nav_written++;
            iter = 0;
            write_time += elapsed_time(&tv);
        }
    }

    if (debug) {
        fprintf(stderr,
                "[debug] read=%.3fms  fft=%.3fms  write=%.3fms\n",
                read_time, fft_time, write_time);
    }

    // cleanup
    if (finp && finp != stdin) fclose(finp);
    if (fout_ac && fout_ac != stdout) fclose(fout_ac);
    if (fout_cc && fout_cc != stdout) fclose(fout_cc);

    for (int i = 0; i < ninp; i++) {
        cudaFree(inp_buf[i]);
        cudaFree(ft_buf[i]);
    }
    free(inp_buf);
    free(ft_buf);

    for (int t = 0; t < MAX_THREADS; t++) {
        if (fftplan[t].doneplan) {
            cufftHandle *plans = fftplan[t].plans;
            for (int j = 0; j < ninp; j++) {
                CUFFT_CHECK(cufftDestroy(plans[j]));
            }
            free(plans);
        }
    }

    for (int t = 0; t < MAX_THREADS; t++) {
        if (reader[t].buffer) free(reader[t].buffer);
    }

    return 0;
}

void print_usage(char *const argv[]) {
    fprintf(stderr,
        "Usage:\n  %s [options]\n"
        "  -p type   A=auto, C=cross, B=both (default %c)\n"
        "  -c num    channels (default %d)\n"
        "  -n num    inputs   (default %d)\n"
        "  -a num    averages before write (default %d)\n"
        "  -i file   input ('-'=stdin)\n"
        "  -o file   output('-'=stdout)\n"
        "  -s num    skip initial spectra\n"
        "  -d        debug\n",
        argv ? argv[0] : "program",
        prod_type, nchan, ninp, naver);
    exit(EXIT_FAILURE);
}

void parse_cmdline(int argc, char *const argv[], const char *optstring) {
    int c;
    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
        case 'c': nchan       = atoi(optarg);       break;
        case 'n': ninp        = atoi(optarg);       break;
        case 'a': naver       = atoi(optarg);       break;
        case 'p': prod_type   = toupper(optarg[0]); break;
        case 'i': infilename  = optarg;             break;
        case 'o': outfilename = optarg;             break;
        case 's': nskip       = atoi(optarg);       break;
        case 'd': debug       = 1;                  break;
        default:  print_usage(argv);
        }
    }
}

void openFiles(const char *infile, const char *outfile, int prod_type,
              FILE **fin, FILE **fout_ac, FILE **fout_cc) {
    char tmp[FILENAME_MAX];
    if (strcmp(infile, "-") == 0) {
        *fin = stdin;
    } else {
        *fin = fopen(infile, "rb");
        if (!*fin) { perror("fopen input"); exit(EXIT_FAILURE); }
    }

    *fout_ac = NULL;
    *fout_cc = NULL;
    if ((prod_type=='A' || prod_type=='B')) {
        if (strcmp(outfile, "-") == 0) {
            *fout_ac = stdout;
        } else {
            snprintf(tmp, sizeof(tmp), "%s.LACSPC", outfile);
            *fout_ac = fopen(tmp, "wb");
            if (!*fout_ac) { perror("fopen auto"); exit(EXIT_FAILURE); }
        }
    }
    if ((prod_type=='C' || prod_type=='B')) {
        if (strcmp(outfile, "-") == 0) {
            *fout_cc = stdout;
        } else {
            snprintf(tmp, sizeof(tmp), "%s.LCCSPC", outfile);
            *fout_cc = fopen(tmp, "wb");
            if (!*fout_cc) { perror("fopen cross"); exit(EXIT_FAILURE); }
        }
    }
}

float elapsed_time(struct timeval *start) {
    struct timeval now;
    gettimeofday(&now, NULL);
    return (now.tv_sec - start->tv_sec)*1e3f
         + (now.tv_usec - start->tv_usec)*1e-3f;
}

int readData(datareader_t *r, int nchan, int ninp,
             FILE *fpin, cufftComplex **inp_buf) {
    if (!r->init) {
        r->ntoread = ninp * BYTES_PER_SAMPLE * nchan;
        r->buffer  = (unsigned char*)malloc(r->ntoread);
        if (!r->buffer) { perror("malloc"); exit(EXIT_FAILURE); }
        r->init = 1;
        if (debug) fprintf(stderr, "[debug] Read buffer %d bytes\n", r->ntoread);
    }

    size_t nread = fread(r->buffer, 1, r->ntoread, fpin);
    if (nread == 0) return 1;  // EOF
    if (nread < (size_t)r->ntoread) {
        fprintf(stderr, "Warning: partial frame read %zu of %d bytes\n",
                nread, r->ntoread);
    }

    for (int inp = 0; inp < ninp; inp++) {
        for (int ch = 0; ch < nchan; ch++) {
            size_t pos = BYTES_PER_SAMPLE * (ch * ninp + inp);
            if (pos + 1 >= nread) break;
            uint8_t lo = r->buffer[pos];
            uint8_t hi = r->buffer[pos+1];
            int16_t sample = (int16_t)((hi << 8) | lo);
            float   realv  = (float)sample / 16.0f;
            inp_buf[inp][ch] = make_cuFloatComplex(realv, 0.0f);
        }
    }
    return 0;
}

void do_FFT_fftw(cufftplan *plan, int nchan, int ninp,
                 cufftComplex **inp_buf, cufftComplex **ft_buf) {
    if (!plan->doneplan) {
        plan->plans = (cufftHandle*)calloc(ninp, sizeof(cufftHandle));
        if (!plan->plans) { perror("calloc plans"); exit(EXIT_FAILURE); }
        for (int i = 0; i < ninp; i++) {
            cufftResult result = cufftPlan1d(&plan->plans[i], nchan, CUFFT_C2C, 1);
            // debug
            if (result != CUFFT_SUCCESS) {
                fprintf(stderr, "Error: cufftPlan1d failed for input %d with error %d\n", i, result);
                exit(EXIT_FAILURE);
            }
        }
        plan->doneplan = 1;
    }
    for (int i = 0; i < ninp; i++) {
        CUFFT_CHECK(cufftExecC2C(plan->plans[i], inp_buf[i], ft_buf[i], CUFFT_FORWARD));
    }
}

void writeOutput(FILE *fout_ac, FILE *fout_cc, int ninp,
                 int nchan, int naver, int prod_type,
                 cufftComplex **buf, float normaliser) {
    if (fout_ac) {
        fprintf(fout_ac, "FFT-only run completed for %d inputs, %d channels\n", ninp, nchan);
    }
    if (fout_cc) {
        fprintf(fout_cc, "FFT-only run completed for %d inputs, %d channels\n", ninp, nchan);
    }
}
