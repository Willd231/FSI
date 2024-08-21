/******************************************************
This is a simple software correlator for complex input data. The input is a stream of data samples (2 bytes)
with one group of bytes for each input per time sample. I.e. for 4 inputs, at time t,
the input would be:
i0i1i2i3 for time t0, then i0i1i2i3 for time t1 etc.
The output is a file of binary float data, real or complex, with adjacent values being the
spectral channels of the correlation products. I.e for channel c, product p the output would be
c0c1...cnchan-1 for p then
c0c1...cnchan-1 for p+1 etc.

Author: Randall Wayth, Feb 2008.

Requires:
- fftw3, including single precision libraries

to compile:
gcc -Wall -O3 -D_FILE_OFFSET_BITS=64 -o corr_cpu corr_cpu.c -lfftw3f -lm

O3 optimisation gives the fastest code, but is impossible to debug with gdb since 
the compiler inlines everything.
Switch to -O -g optimisation or no optimisation for debugging.

Use:
start the program with no arguments for a summary of command line options.
Command-line options control the number of inputs, number of freq channels,
type of correlation product to write (auto, cross or both),
the number of averages before writing output and the input and output file names.
(stdin,stdout can also be used in some cases)
Debugging/timing output can also be generated with the command line option '-d'.

The code reads blocks of ninp*nchan complex samples and re-shuffles them into ninp arrays of nchan complex floats.
(for nchan spectral channels, not including the total power term, one needs nchan complex values to FFT.)
Each input array is then FFT'd into a complex spectrum. Correlation products (auto and/or cross)
are then generated for each of these spectra and accumulated.

After the specified number of spectra have been added into
the accumulations (the number of averages), they are written to output after being
averaged and normalised for the FFT. In this way, one or more averages can be written per output file.
If the end of the input is reached and nothing has yet been written, one average is written regardless
of the command-line '-a' argument. If one or more averages have already been written and the end of
the input has been reached, then partially accumulated data is discarded.

By default, two output files are created with the suffixes ".LACSPC" and ".LCCSPC" for the
auto and cross correlation products respectively. If only one type of correlation product is generated,
then stdout can be used.
The output format of autocorrelation is binary single precision floats,
one spectrum for each correlation product, per average.
The output is ordered by channel, then product, then average. I.e. for 4 output products with 512 channels,
the data will be channel1, channel2... channel512 all for product 1, then the same for
channel 2 etc. Then the whole thing is repeated again if there are more than 1 averages.
The output format of cross correlation is binary single precision complexes,
one spectrum for each correlation product, per average. Same idea with data ordering.

It is advisable to use input files that are an integer multiple of ninp*nchan samples long, otherwise there
will be unused data at the end of the file. If more than one average put input file is desired,
it is also advisable to use files that are ninp*nchan*naver samples long so that all data is used in an output average.

How many averages do I need to average for t seconds?
- for Nyquist sampled byte data with bandwidth B Hz, there are B*n_inputs samples per second.
- for n_channels output, we consume n_chan*n_input samples per input batch.
- therefore B/nchan (batches/sec) gives the number of averages required for 1 second of data.

Input data size:
this code was written for MWA 8T/32T data with 2 byte complex input samples. The input is 5real+5imag
bits in the lowest bits of the 16-bit word. The 5-bit numbers are two's complement integers.

******************************************************/

/*

Modified to process CARSE data. Based on Will Delligner's program for reading CARSE data.

Anish July 08, 2024

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <ctype.h>
#include <complex.h> /* include this before fftw3 */
#include <fftw3.h>

#define MAX_THREADS 1
#define BYTES_PER_SAMPLE 2
#define BITS_PER_NUM 5
#define TWOCOMP_MAXUINT   (1<<(BITS_PER_NUM))    // = 32 for 5 bits
#define TWOCOMP_ZEROPOINT (1<<(BITS_PER_NUM-1))  // = 16 for 5 bits
#define DECODER_MASK ((1<<(2*BITS_PER_NUM)) -1)

typedef struct _fftplan {
    fftwf_plan *plans;
    int doneplan;
} fftplan_t;

typedef struct _datareader {
    int init,ntoread, offset; /* Added offset to skip the first 64 bytes or 32 int_16 values. Anish July 5m 2024*/
    unsigned char *buffer;  
    fftwf_complex *decoder;
} datareader_t;

/* function prototypes */
void print_usage(char * const argv[]);
void parse_cmdline(int argc, char * const argv[], const char *optstring);
int readData(datareader_t *reader,int nchan,int ninp,FILE *fpin,fftwf_complex **inp_buf);
int openFiles(char *infilename, char *outfilename, int prod_type, FILE **fin, FILE **fout_ac, FILE **fout_cc);
void do_FFT_fftw(fftplan_t *plan, int nchan, int ninp, fftwf_complex **inp_buf, fftwf_complex **ft_buf);
void do_CMAC(const int nchan,const int ninp,const int prod_type, fftwf_complex **ft_buf, fftwf_complex **corr_buf);
void writeOutput(FILE *fout_ac,FILE *fout_cc,int ninp, int nchan, int naver, int prod_type,fftwf_complex **buf, float normaliser);
float elapsed_time(struct timeval *start);

/* global vars */
int nchan=1024;      /* number of channels in output spectra - Anish July 5, 2024 */
int ninp=4;         /* input of inputs */
int debug=0;        /* turn debugging info on */
int naver=99999;    /* number of input spectra to accumulate before writing an average */
int prod_type='B';  /* correlation product type: B: both, C: cross, A: auto */
char *infilename=NULL,*outfilename=NULL;
int nskip=0;        /* skip nchan * nskip data from the begining of the file  - set to 0 Anish July 5, 2024*/

int main(int argc, char * const argv[]) {
    int filedone=0,res=0,i,ncorr,iter=0,nav_written=0;
    FILE *finp=NULL,*fout_ac=NULL,*fout_cc=NULL;
    char optstring[]="dc:i:o:n:a:p:s:";
    float normaliser=1.0;
    float read_time=0, fft_time=0, cmac_time=0, write_time=0;
    fftwf_complex **inp_buf=NULL, **ft_buf=NULL,**corr_buf=NULL;
    struct timeval thetime;
    fftplan_t fftplan[MAX_THREADS];
    datareader_t reader[MAX_THREADS];

    /* initialise */
    for(i=0;i<MAX_THREADS;i++) {
        fftplan[i].doneplan=0;
        fftplan[i].plans=NULL;
        reader[i].init=0;
        reader[i].buffer=NULL;
        reader[i].decoder=NULL;
    }

    /* process command line args */
    if (argc <2) print_usage(argv);
    parse_cmdline(argc,argv,optstring);
    ncorr = ninp*(ninp+1)/2;
    if (debug) {
        fprintf(stderr,"Num inp:\t%d. Num corr products: %d\n",ninp,ncorr);
        fprintf(stderr,"Num chan:\t%d\n",nchan);
        fprintf(stderr,"Correlation product type:\t%c\n",prod_type);
        fprintf(stderr,"infile: \t%s\n",infilename);
        fprintf(stderr,"outfile:\t%s\n",outfilename);
    }
    
    /* open input and output files */
    openFiles(infilename,outfilename,prod_type, &finp,&fout_ac,&fout_cc);

    /* allocate buffers */
    /* for input and FFT results */
    inp_buf = (fftwf_complex **) calloc(ninp,sizeof(fftwf_complex *));
    ft_buf  = (fftwf_complex **) calloc(ninp,sizeof(fftwf_complex *));
    if (inp_buf == NULL || ft_buf==NULL) {
        fprintf(stderr,"malloc failed\n");
        exit(1);
    }
    for (i=0; i< ninp; i++) {
        inp_buf[i]= (fftwf_complex *) fftwf_malloc(nchan*sizeof(fftwf_complex));
        ft_buf[i] = (fftwf_complex *) fftwf_malloc(nchan*sizeof(fftwf_complex));
    }
    /* for correlation products */
    corr_buf = (fftwf_complex **) calloc(ncorr,sizeof(fftwf_complex *));
    for(i=0; i<ncorr;i++) {
        corr_buf[i] = (fftwf_complex *) fftwf_malloc(nchan*sizeof(fftwf_complex));
        /* init to zero  */
        memset(corr_buf[i],'\0',(nchan)*sizeof(fftwf_complex));
    }

    /*skip the first nskip x nchan data
      This is introduced to skip the tone burst during correlation
      Anish, Dec 03, 2010  */
 
    for(i=0; i<nskip; i++) {
        res = readData(reader,nchan, ninp,finp,inp_buf);
    } 

    /* process file */
    while (!filedone) {
    
        /* read time chunk into buffers */
        gettimeofday(&thetime,NULL);
        res = readData(reader,nchan, ninp,finp,inp_buf);
        if (res !=0) {
            filedone=1;
        }
        read_time += elapsed_time(&thetime);
        
        if (!filedone) {
            /* do the FFT */
            gettimeofday(&thetime,NULL);
            do_FFT_fftw(fftplan,nchan,ninp,inp_buf,ft_buf);
            fft_time += elapsed_time(&thetime);
        
            /* do the CMAC */
            gettimeofday(&thetime,NULL);
            do_CMAC(nchan,ninp,prod_type, ft_buf,corr_buf);
            cmac_time += elapsed_time(&thetime);
        
        }
        
        /* write and average if it is time to */
        if ((filedone && nav_written==0) || ++iter == naver) {
            gettimeofday(&thetime,NULL);
            normaliser = 1.0/(nchan*iter);
            writeOutput(fout_ac,fout_cc,ninp,nchan,iter,prod_type,corr_buf,normaliser);
            if(debug) fprintf(stderr,"writing average of %d chunks\n",iter);
            iter=0;
            nav_written++;
            write_time += elapsed_time(&thetime);
        }
    }
    
    if (debug) {
        fprintf(stderr,"wrote %d averages. unused chunks: %d\n",nav_written,iter);
        fprintf(stderr,"Time reading:\t%g ms\n",read_time);
        fprintf(stderr,"Time FFTing:\t%g ms\n",fft_time);
        fprintf(stderr,"Time CMACing:\t%g ms\n",cmac_time);
        fprintf(stderr,"Time writing:\t%g ms\n",write_time);
    }
    
    /* clean up */
    fclose(finp);
    if(fout_ac !=NULL) fclose(fout_ac);
    if(fout_cc !=NULL) fclose(fout_cc);
    for (i=0; i<ninp; i++) {
        if (inp_buf[i]!=NULL) fftwf_free(inp_buf[i]);
        if (ft_buf[i]!=NULL)  fftwf_free(ft_buf[i]);
    }
    free(inp_buf); free(ft_buf);
    return 0;
}


/* write out correlation products.
   Apply a normalisation factor that depends on the FFT length and the number
   of averages so the flux density is the same regardless of the spectral channel width */
void writeOutput(FILE *fout_ac, FILE *fout_cc,int ninp, int nchan, int naver, int prod_type,
                fftwf_complex **buf,float normaliser) {
    int inp1,inp2,cprod=0,chan;
    float *temp_buffer=NULL;
    
    temp_buffer=malloc(sizeof(float)*(nchan));
    
    for(inp1=0; inp1<ninp; inp1++) {
        for (inp2=inp1; inp2<ninp; inp2++) {
            /* make an average by dividing by the number of chunks that went into the total */
            for (chan=0; chan<nchan; chan++) {
                buf[cprod][chan] *= normaliser;
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
                fwrite(buf[cprod],sizeof(fftwf_complex),nchan,fout_cc);
            }

            /* reset the correlation products to zero */
            memset(buf[cprod],'\0',(nchan)*sizeof(fftwf_complex));
            cprod++;
        }
    }
    if (temp_buffer!=NULL) free(temp_buffer);
}


/* accumulate correlation products */
void do_CMAC(const int nchan,const int ninp,const int prod_type, fftwf_complex **ft_buf, fftwf_complex **corr_buf) {
    int inp1,inp2,cprod=0;
    fftwf_complex *ftinp1,*ftinp2,*cbuf;
    register int chan;
    
    for(inp1=0; inp1<ninp; inp1++) {
        ftinp1 = ft_buf[inp1];
        for(inp2=inp1; inp2<ninp; inp2++) {
            ftinp2 = ft_buf[inp2];
            cbuf = corr_buf[cprod];
            if(prod_type=='B' || ((prod_type=='C' && inp1!=inp2) || (prod_type=='A' && inp1==inp2))) {
                for(chan=0;chan<nchan;chan++) {
                    cbuf[chan] += ftinp1[chan]*conjf(ftinp2[chan]);
                }
            }
            cprod++;         
        }
    }    
}


/* do the FFT */
void do_FFT_fftw(fftplan_t *fftplan, int nchan, int ninp, fftwf_complex **inp_buf, fftwf_complex **ft_buf) {   
    int i;
    
    /* make the FFTW execution plans */
    /* since each input and output buffer is different, we can't re-use the same plan, which is clunky
       but there's not much we can do about it, short of moving data around unnecesarily. We create one
       plan for each input stream, just once, and use it many times */
    if (!fftplan->doneplan) {
        fftplan->plans = (fftwf_plan *) calloc(ninp,sizeof(fftwf_plan));
        for (i=0; i<ninp; i++) {
            fftplan->plans[i] = fftwf_plan_dft_1d(nchan, inp_buf[i], ft_buf[i],FFTW_FORWARD, FFTW_ESTIMATE);
        }
        fftplan->doneplan=1;
    }
    
    for (i=0; i<ninp; i++) {
        fftwf_execute(fftplan->plans[i]);
    }
}


/* read in a batch of input data, n_channels*n_inputs complex samples.
   each sample has BITS_PER_NUM real and BITS_PER_NUM imaginary bits.
   convert to complex and pack into the input buffers for later FFT */
int readData(datareader_t *reader,int nchan,int ninp,FILE *fpin,fftwf_complex **inp_buf) {

    int inp,chan,nread,ind;
//    unsigned short *sample; Anish
    __int16 *sample;

    if (!reader->init) {
        int decoder_buf_size=(1<<(2*BITS_PER_NUM)),im,re,index=0,rval,ival;

        reader->init=1;
        /* reader->ntoread = ninp*BYTES_PER_SAMPLE*nchan;  number of bytes to read per batch Anish July 5, 2024*/
        reader->ntoread = 8256;  /* number of bytes to read per batch is set as packetsize Anish July 5, 2024*/
        reader->offset = 32;  /* 32 int_16 values to be skipped Anish July 5, 2024*/
        reader->buffer = (unsigned char *) malloc(reader->ntoread);
        reader->decoder = malloc(sizeof(fftwf_complex)*decoder_buf_size);
       
//The decoder is not needed for PAF
//Anish, Nov 9, 2010
 
        /* initialise the decoder lookup table for 5+5bit signed integer in twos complement format 
        for(re=0;re< TWOCOMP_MAXUINT ;re++) {
            rval = re >= TWOCOMP_ZEROPOINT ? re-TWOCOMP_MAXUINT : re;
            for(im=0;im < TWOCOMP_MAXUINT;im++) {
            
                ival = im >= TWOCOMP_ZEROPOINT ? im-TWOCOMP_MAXUINT : im;
            
                reader->decoder[index] = rval + I*ival;
                //printf("decoder: (%g,%g)\n",creal(reader->decoder[index]),cimag(reader->decoder[index]));
                index++;
            }
        }*/
        
        if (debug) {
            fprintf(stderr,"size of read buffer: %d bytes\n",reader->ntoread);
            fprintf(stderr,"size of decoder lookup: %d elements\n",decoder_buf_size);
        }
    }

    nread = fread(reader->buffer,1,reader->ntoread,fpin);
    if(nread < reader->ntoread) return 1;
 
    for(inp=0; inp<ninp; inp++) {
        for(chan=0; chan<nchan; chan++) {
//            sample = ((unsigned short *)reader->buffer)+(ninp*chan + inp); Anish
            sample = ((__int16 *)reader->buffer)+(ninp*chan + inp + reader->offset);
//            ind = (*sample) & DECODER_MASK; Anish
//            inp_buf[inp][chan] = reader->decoder[ind]; Anish
//            inp_buf[inp][chan] = (*sample)/16 + I*0; Anish July 5, 2024
            inp_buf[inp][chan] = (*sample)/4 + I*0; /* 16 to 14 bit number convertion Anish July 5, 2024 */
//            printf("sample: %d. Val: (%g,%g)\n",(unsigned int) *sample, creal(inp_buf[inp][chan]),cimag(inp_buf[inp][chan]));
//            printf("sample: %d. Val: (%g,%g)\n",(int) *sample, creal(inp_buf[inp][chan]),cimag(inp_buf[inp][chan]));
        }
    }

    return 0;
}


/* open the input and output files */
int openFiles(char *infilename, char *outfilename, int prod_type, FILE **fin, FILE **fout_ac, FILE **fout_cc) {
    char tempfilename[FILENAME_MAX];
    
    if (infilename == NULL) {
        fprintf(stderr,"No input file specified\n");
        exit(1);
    }
    if (outfilename == NULL) {
        fprintf(stderr,"No output file specified\n");
        exit(1);
    }
    
    /* sanity check: can only use stdout for one type of output */
    if((prod_type=='B') && strcmp(outfilename,"-")==0) {
        fprintf(stderr,"Can only use stdout for either auto or cross correlations, not both\n");
        exit(1);
    }
    
    /* check for special file name: "-", which indicates to use stdin/stdout */
    if (strcmp(infilename,"-")==0) {
        *fin = stdin;
    } else {
        *fin = fopen(infilename,"r");
        if (*fin ==NULL) {
            fprintf(stderr,"failed to open input file name: <%s>\n",infilename);
            exit(1);
        }        
    }
    
    if ((prod_type=='A') && strcmp(outfilename,"-")==0) {
        *fout_ac = stdout;
    } else if ((prod_type=='C') && strcmp(outfilename,"-")==0) {
        *fout_cc = stdout;
    } else {
        if (prod_type=='A' || prod_type=='B') {
            strncpy(tempfilename,outfilename,FILENAME_MAX-8);
            strcat(tempfilename,".LACSPC");
            *fout_ac = fopen(tempfilename,"w");
            if (*fout_ac ==NULL) {
                fprintf(stderr,"failed to open output file name: <%s>\n",tempfilename);
                exit(1);
            }
        } 
        if (prod_type=='C' || prod_type=='B') {
            strncpy(tempfilename,outfilename,FILENAME_MAX-8);
            strcat(tempfilename,".LCCSPC");
            *fout_cc = fopen(tempfilename,"w");
            if (*fout_cc ==NULL) {
                fprintf(stderr,"failed to open output file name: <%s>\n",tempfilename);
                exit(1);
            }
        } 
    }
    
    return 0;
}


void parse_cmdline(int argc, char * const argv[], const char *optstring) {
    int c;
    
    while ((c=getopt(argc,argv,optstring)) != -1) {
        switch(c) {
            case 'c':
                nchan = atoi(optarg);
                if (nchan <=0 || nchan > 65536) {
                    fprintf(stderr,"bad number of channels: %d\n",nchan);
                    print_usage(argv);
                }
                break;
            case 'n':
                ninp = atoi(optarg);
                if (ninp <=0 || ninp > 128) {
                    fprintf(stderr,"bad number of inputs: %d\n",ninp);
                    print_usage(argv);
                }
                break;
            case 'a':
                naver = atoi(optarg);
                if (naver <=0 || naver > 9999999) {
                    fprintf(stderr,"bad number of averages: %d\n",naver);
                    print_usage(argv);
                }
                break;
            case 'p':
                prod_type = toupper(optarg[0]);
                if (prod_type!='A' && prod_type !='B' && prod_type != 'C') {
                    fprintf(stderr,"bad correlation product type: %c\n",prod_type);
                    print_usage(argv);
                }
                break;
            case 'i':
                infilename=optarg;
                break;
            case 'd':
                debug=1;
                break;
            case 'o':
                outfilename=optarg;
                break;
            case 's':
                nskip=atoi(optarg);
                break;
            default:
                fprintf(stderr,"unknown option %c\n",c);
                print_usage(argv);
        }
    }
}


void print_usage(char * const argv[]) {
    fprintf(stderr,"Usage:\n%s [options]\n",argv[0]);
    fprintf(stderr,"\t-p type\t\tspecify correlation product type(s). A: auto, C: cross, B: both. default: %c\n",prod_type);
    fprintf(stderr,"\t-c num\t\tspecify number of freq channels. default: %d\n",nchan);
    fprintf(stderr,"\t-n num\t\tspecify number of input streams. detault: %d\n",ninp);
    fprintf(stderr,"\t-a num\t\tspecify min number of averages before output. Default: %d\n",naver);
    fprintf(stderr,"\t-i filename\tinput file name. use '-' for stdin\n");
    fprintf(stderr,"\t-o filename\toutput file name. use '-' for stdout\n");
    fprintf(stderr,"\t-s num\tnum of nchan data to be skipped from the beginning of the data file\n"); //Anish Dec 03, 2010
    fprintf(stderr,"\t-d         \twrite debug and runtime info to stderr\n");
    exit(0);
}


/* returns the elapsed wall-clock time, in ms, since start (without resetting start) */
float elapsed_time(struct timeval *start){
    struct timeval now;
    gettimeofday(&now,NULL);
    return 1.e3f*(float)(now.tv_sec-start->tv_sec) +
        1.e-3f*(float)(now.tv_usec-start->tv_usec);
}
