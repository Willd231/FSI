// Hampel despike across frequency per spectrum, preserving file layout.
// Layout per record: float32 ts, float32 freq[512], float32 autospec[4][512]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <math.h>

#define NINP  4
#define NCHAN 512
#define EPS   1e-8f

typedef struct {
    float ts;
    float freq[NCHAN];
    float spec[NINP][NCHAN];
} Record;

static void must_read(void *p, size_t sz, size_t n, FILE *fp, const char *w){
    if (fread(p, sz, n, fp) != n){ fprintf(stderr,"Read failed: %s\n", w); exit(1); }
}
static void must_write(const void *p, size_t sz, size_t n, FILE *fp, const char *w){
    if (fwrite(p, sz, n, fp) != n){ fprintf(stderr,"Write failed: %s\n", w); exit(1); }
}
static size_t rec_bytes(void){ return sizeof(float) * (1 + NCHAN + NINP * NCHAN); }

static size_t full_records(const char *path, size_t *tail){
    struct stat st; if (stat(path, &st)!=0){ perror("stat"); exit(1); }
    size_t rb = rec_bytes();
    size_t n  = (size_t)(st.st_size / rb);
    if (tail) *tail = (size_t)(st.st_size - (long long)n * rb);
    return n;
}

/* insertion sort for tiny windows */
static inline void isort(float *a, int n){
    for (int i=1;i<n;++i){
        float x=a[i]; int j=i-1;
        while (j>=0 && a[j]>x){ a[j+1]=a[j]; --j; }
        a[j+1]=x;
    }
}
static inline float median_sorted(const float *a, int n){ return a[n/2]; }
static inline float median_copy(const float *x, int n){
    float tmp[17]; /* win up to 17; bump if you want bigger */
    for (int i=0;i<n;++i) tmp[i]=x[i];
    isort(tmp,n);
    return median_sorted(tmp,n);
}

/* Hampel filter (freq axis) for a single channel array of length NCHAN */
static void hampel_freq(float *arr, int n, int win, float k){
    int r = win/2;
    float wbuf[17];  /* data window */
    float abuf[17];  /* abs dev window */

    for (int i=0;i<n;++i){
        /* build window with edge-replication */
        int w=0;
        for (int off=-r; off<=r; ++off){
            int idx = i + off;
            if (idx < 0) idx = 0;
            if (idx >= n) idx = n-1;
            wbuf[w++] = arr[idx];
        }
        float med = median_copy(wbuf, win);
        for (int j=0;j<win;++j) abuf[j] = fabsf(wbuf[j] - med);
        float mad = median_copy(abuf, win);
        float sigma = 1.4826f * mad + EPS;

        if (fabsf(arr[i] - med) > k * sigma){
            /* replace hard outliers with the local median */
            arr[i] = med;
        }
    }
}

static void usage(const char *p){
    fprintf(stderr,
      "Usage: %s --win <odd>=7 --k <float>=4.0 <in.dat> <out.dat>\n"
      "Applies Hampel despiking across frequency per spectrum (all 4 inputs).\n", p);
    exit(1);
}

int main(int argc, char **argv){
    if (argc < 3) usage(argv[0]);

    int win = 7;        /* odd */
    float k = 4.0f;
    const char *inpath=NULL, *outpath=NULL;

    for (int i=1;i<argc;i++){
        if (!strcmp(argv[i],"--win") && i+1<argc){ win = atoi(argv[++i]); }
        else if (!strcmp(argv[i],"--k") && i+1<argc){ k = (float)atof(argv[++i]); }
        else if (!inpath){ inpath = argv[i]; }
        else if (!outpath){ outpath = argv[i]; }
        else usage(argv[0]);
    }

    if (win < 3 || (win & 1) == 0 || win > 17){
        fprintf(stderr,"win must be odd in [3..17] (got %d)\n", win);
        return 1;
    }
    if (!inpath || !outpath) usage(argv[0]);

    size_t tail=0, nspec = full_records(inpath, &tail);
    if (nspec==0){ fprintf(stderr,"No full records.\n"); return 1; }
    if (tail) fprintf(stderr,"Warning: ignoring %zu trailing bytes (partial record)\n", tail);

    FILE *fin=fopen(inpath,"rb"); if(!fin){ perror("fopen in"); return 1; }
    FILE *fout=fopen(outpath,"wb"); if(!fout){ perror("fopen out"); fclose(fin); return 1; }

    Record rec;
    for (size_t s=0; s<nspec; ++s){
        must_read(&rec.ts,   sizeof(float), 1,             fin,  "ts");
        must_read(&rec.freq, sizeof(float), NCHAN,         fin,  "freq");
        must_read(&rec.spec, sizeof(float), NINP*NCHAN,    fin,  "spec");

        /* per-input despike across frequency bins */
        for (int inp=0; inp<NINP; ++inp){
            hampel_freq(rec.spec[inp], NCHAN, win, k);
        }

        must_write(&rec.ts,   sizeof(float), 1,             fout, "ts out");
        must_write(&rec.freq, sizeof(float), NCHAN,         fout, "freq out");
        must_write(&rec.spec, sizeof(float), NINP*NCHAN,    fout, "spec out");
    }

    fclose(fin); fclose(fout);
    fprintf(stderr,"Done: %s → %s (Hampel freq, win=%d, k=%.2f)\n", inpath, outpath, win, k);
    return 0;
}
