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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult _status = (call);                                           \
        if (_status != CUFFT_SUCCESS) {                                         \
            fprintf(stderr, "cuFFT Error %s:%d: %d\n",                      \
                    __FILE__, __LINE__, (int)_status);                         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
    
typedef struct {
    cufftHandle *plans;
    int          doneplan;
} cufftplan;

typedef struct {
    int            init;
    int            ntoread;
    unsigned char *buffer;
} datareader_t;

/* Global parameters */
int  nchan        = 128;
int  ninp         = 4;
int  debug        = 0;
int  naver        = 99999;
int  prod_type    = 'B';
char *infilename  = NULL;
char *outfilename = NULL;
int  nskip        = 0;

/* Function prototypes */
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
                   int nchan, int iter, int prod_type,
                   cufftComplex **buf, float normaliser);

int main(int argc, char *const argv[]) {
    int             filedone = 0;
    int             res = 0;
    int             iter = 0;
    int             nav_written = 0;
    int             ncorr = ninp * (ninp + 1) / 2;
    FILE           *finp = NULL, *fout_ac = NULL, *fout_cc = NULL;
    float           normaliser = 1.0f;
    float           read_time = 0, fft_time = 0, write_time = 0;
    cufftComplex  **inp_buf = NULL, **ft_buf = NULL;
    struct timeval  tv;
    cufftplan       fftplan[MAX_THREADS];
    datareader_t    reader[MAX_THREADS];
    long long       loop_count = 0;

    // Initialize plan and reader structures
    for (int t = 0; t < MAX_THREADS; t++) {
        fftplan[t].doneplan = 0;
        fftplan[t].plans    = NULL;
        reader[t].init      = 0;
        reader[t].buffer    = NULL;
    }

    // Parse command line
    if (argc < 2) print_usage(argv);
    parse_cmdline(argc, argv, "dc:i:o:n:a:p:s:");

    if (!infilename || !outfilename) {
        fprintf(stderr, "Error: must specify -i and -o\n");
        print_usage(argv);
    }

    openFiles(infilename, outfilename, prod_type,
              &finp, &fout_ac, &fout_cc);

    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));
    if (debug) fprintf(stderr, "CUDA runtime init\n");

    // Allocate host-managed buffers
    inp_buf = (cufftComplex**)calloc(ninp, sizeof(cufftComplex*));
    ft_buf  = (cufftComplex**)calloc(ninp, sizeof(cufftComplex*));
    if (!inp_buf || !ft_buf) { perror("calloc"); exit(EXIT_FAILURE); }

    for (int i = 0; i < ninp; i++) {
        CUDA_CHECK(cudaMallocManaged((void**)&inp_buf[i], nchan * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMallocManaged((void**)&ft_buf[i],  nchan * sizeof(cufftComplex)));
    }

    // Skip initial spectra if needed
    for (int i = 0; i < nskip; i++) {
        readData(&reader[0], nchan, ninp, finp, inp_buf);
    }

    // Main processing loop
    while (!filedone) {
        gettimeofday(&tv, NULL);
        res = readData(&reader[0], nchan, ninp, finp, inp_buf);
        if (debug && res)
            fprintf(stderr, "EOF reached @ iteration %lld\n", loop_count);
        read_time += elapsed_time(&tv);
        if (res) filedone = 1;

        if (!filedone) {
            if (debug && (loop_count % 1000) == 0)
                fprintf(stderr, "db  fft iteration %lld\n", loop_count);
            gettimeofday(&tv, NULL);
            do_FFT_fftw(&fftplan[0], nchan, ninp, inp_buf, ft_buf);
            CUDA_CHECK(cudaDeviceSynchronize());
            fft_time += elapsed_time(&tv);
        }

        if ((filedone && !nav_written) || ++iter == naver) {
            if (debug) fprintf(stderr, "Calling writeOutput: iter=%d, filedone=%d\n", iter, filedone);
            gettimeofday(&tv, NULL);
            normaliser = 1.0f / (nchan * iter);
            writeOutput(fout_ac, fout_cc, ninp, nchan, iter, prod_type, ft_buf, normaliser);

            //may need to remove this
            if (fout_ac) fflush(fout_ac);
            if (fout_cc) fflush(fout_cc);
            //---------------

            nav_written++;
            iter = 0;
            write_time += elapsed_time(&tv);
        }

        loop_count++;
    }

    // Final debug summary
    if (debug) {
        fprintf(stderr,
                "read=%.3fms  fft=%.3fms  write=%.3fms\n",
                read_time, fft_time, write_time);
    }

    // Cleanup
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
            for (int j = 0; j < ninp; j++) {
                CUFFT_CHECK(cufftDestroy(fftplan[t].plans[j]));
            }
            free(fftplan[t].plans);
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
        argv[0], prod_type, nchan, ninp, naver);
    exit(EXIT_FAILURE);
}

void parse_cmdline(int argc, char *const argv[], const char *optstring) {
    int c;
    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
            case 'c': nchan       = atoi(optarg);       break;
            case 'n': ninp        = atoi(optarg);       break;
            case 'a': naver       = atoi(optarg);       break;
            case 'p': 
                prod_type = toupper(optarg[0]); 
                if (prod_type != 'A' && prod_type != 'C' && prod_type != 'B') {
                    fprintf(stderr, "Error: Invalid product type '%c'. Must be A, C, or B.\n", prod_type);
                    exit(EXIT_FAILURE);
                }
                break;
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

    *fout_ac = NULL; *fout_cc = NULL;
    if (prod_type=='A' || prod_type=='B') {
        if (strcmp(outfile, "-") == 0) *fout_ac = stdout;
        else {
            snprintf(tmp, sizeof(tmp), "%s.LACSPC", outfile);
            *fout_ac = fopen(tmp, "wb"); 
            if (!*fout_ac) { 
                perror("fopen auto"); 
                exit(EXIT_FAILURE); 
            }
        }
    }
    if (prod_type=='C' || prod_type=='B') {
        if (strcmp(outfile, "-") == 0) *fout_cc = stdout;
        else {
            snprintf(tmp, sizeof(tmp), "%s.LCCSPC", outfile);
            *fout_cc = fopen(tmp, "wb"); 
            if (!*fout_cc) { 
                perror("fopen cross"); 
                exit(EXIT_FAILURE); 
            }
        }
    }
}

float elapsed_time(struct timeval *start) {
    struct timeval now; gettimeofday(&now, NULL);
    return (now.tv_sec - start->tv_sec)*1e3f + (now.tv_usec - start->tv_usec)*1e-3f;
}

int readData(datareader_t *r, int nchan, int ninp,
             FILE *fpin, cufftComplex **inp_buf) {
    if (!r->init) {
        r->ntoread = ninp * BYTES_PER_SAMPLE * nchan;
        r->buffer  =(unsigned char*)malloc(r->ntoread);
        if (!r->buffer) { perror("malloc"); exit(EXIT_FAILURE); }
        r->init = 1;
    }
    size_t nread = fread(r->buffer, 1, r->ntoread, fpin);
    if (nread == 0) return 1;
    if (nread < (size_t)r->ntoread)
        fprintf(stderr, "Warning: partial read %zu of %d bytes\n", nread, r->ntoread);
    for (int inp=0; inp<ninp; inp++) for (int ch=0; ch<nchan; ch++){
        size_t pos = 2*(ch*ninp+inp);
        if (pos+1>=nread) break;
        uint8_t lo=r->buffer[pos], hi=r->buffer[pos+1];
        int16_t s=(int16_t)((hi<<8)|lo);
        inp_buf[inp][ch]=make_cuFloatComplex((float)s/16.0f,0.0f);
    }
    return 0;
}

void do_FFT_fftw(cufftplan *plan, int nchan, int ninp,
                 cufftComplex **inp_buf, cufftComplex **ft_buf){
    if (!plan->doneplan){
        plan->plans = (cufftHandle*)calloc(ninp, sizeof(cufftHandle));
        if (!plan->plans){ perror("calloc"); exit(EXIT_FAILURE);}\
        for(int i=0;i<ninp;i++) CUFFT_CHECK(cufftPlan1d(&plan->plans[i],nchan,CUFFT_C2C,1));
        plan->doneplan=1;
    }
    for(int i=0;i<ninp;i++) CUFFT_CHECK(cufftExecC2C(plan->plans[i], inp_buf[i], ft_buf[i], CUFFT_FORWARD));
}

void writeOutput(FILE *fout_ac, FILE *fout_cc, int ninp, int nchan, int iter, int prod_type,cufftComplex **buf, float normaliser){


   float * temp_buffer = NULL; 
   temp_buffer=(float*)malloc(sizeof(float)*(nchan));
   int inp1 = 0; 
   int inp2 = 0;
   int chan = 0; 
   int cprod = 0; 
    
    for(inp1=0; inp1<ninp; inp1++) {
        for (inp2=inp1; inp2<ninp; inp2++) {
            /* make an average by dividing by the number of chunks that went into the total */
            for (chan=0; chan<nchan; chan++) {
                buf[cprod][chan] = buf[cprod][chan] *  normaliser;
                /* convert the autocorrelation numbers into floats, since the imag parts will be zero*/
                if (inp1==inp2 && (prod_type == 'A' || prod_type=='B')){
                
                    temp_buffer[chan] = creal(buf[cprod][chan]);
                }
            }
            if(inp1==inp2 && (prod_type == 'A' || prod_type=='B')) {
                /* write the auto correlation product */
                fwrite(temp_buffer,sizeof(float),nchan,fout_ac);
            }
            if(inp1!=inp2 && (prod_type == 'C' || prod_type=='B')) {
                /* write the cross correlation product */
                fwrite(buf[cprod],sizeof(cufftComplex),nchan,fout_cc);
            }

            /* reset the correlation products to zero */
            memset(buf[cprod],'\0',(nchan)*sizeof(cufftComplex));
            cprod++;
        }
    }
    if (temp_buffer!=NULL) free(temp_buffer);
}
