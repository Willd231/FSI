// Robust RFI cleaning across frequency per spectrum.
// 1) wide median baseline, 2) Hampel on residual, 3) tiny post median.
// Layout per record: float32 ts, float32 freq[512], float32 spec[4][512]

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

/* insertion sort for tiny windows (<= 127 okay but we’ll cap to 65 for stack) */
static inline void isort(float *a, int n){
    for (int i=1;i<n;++i){ float x=a[i]; int j=i-1; while (j>=0 && a[j]>x){ a[j+1]=a[j]; --j; } a[j+1]=x; }
}
static inline float median_copy(const float *x, int n){
    // window size n is odd and small; copy to stack
    float tmp[129];
    if (n > 129){ fprintf(stderr,"window too large\n"); exit(1); }
    for (int i=0;i<n;++i) tmp[i]=x[i];
    isort(tmp,n);
    return tmp[n/2];
}

/* edge-replicated window gather */
static inline int clampi(int v, int lo, int hi){ if (v<lo) return lo; if (v>hi) return hi; return v; }
static void win_vals(const float *a, int n, int center, int win, float *out){
    int r = win/2, k=0;
    for (int off=-r; off<=r; ++off){
        int idx = clampi(center+off, 0, n-1);
        out[k++] = a[idx];
    }
}

/* stage 1: wide median baseline */
static void baseline_median(const float *x, int n, int win, float *base){
    float buf[129];
    for (int i=0;i<n;++i){
        win_vals(x, n, i, win, buf);
        base[i] = median_copy(buf, win);
    }
}

/* stage 2: Hampel on residual */
static void hampel_residual(float *res, int n, int win, float k){
    float buf[129], abuf[129];
    for (int i=0;i<n;++i){
        win_vals(res, n, i, win, buf);
        float med = median_copy(buf, win);
        for (int j=0;j<win;++j) abuf[j] = fabsf(buf[j]-med);
        float mad = median_copy(abuf, win);
        float sigma = 1.4826f * mad + EPS;
        float d = res[i] - med;
        if (fabsf(d) > k * sigma){
            // soft replacement: pull toward median, not hard clamp (reduces “spiky” look)
            res[i] = med; // or: med + copysignf(k*sigma, d);  // clip instead of replace
        }
    }
}

/* stage 3: tiny post median to smooth jaggies */
static void post_median(float *x, int n, int win){
    if (win <= 1) return;
    float buf[9];
    float *tmp = (float*)malloc(n*sizeof(float));
    if (!tmp){ fprintf(stderr,"oom\n"); exit(1); }
    for (int i=0;i<n;++i){
        win_vals(x, n, i, win, buf);
        tmp[i] = median_copy(buf, win);
    }
    memcpy(x, tmp, n*sizeof(float));
    free(tmp);
}

static void usage(const char *p){
    fprintf(stderr,
      "Usage: %s --base <odd>=51 --hwin <odd>=7 --k <float>=3.8 --post <odd>=3 <in.dat> <out.dat>\n"
      "Stages: wide baseline median, Hampel on residual, tiny post median (all across frequency).\n", p);
    exit(1);
}

int main(int argc, char **argv){
    int base_win = 51;    // wide baseline
    int hwin     = 7;     // Hampel window
    float k      = 3.8f;  // Hampel threshold
    int post_win = 3;     // tiny smoothing
    const char *inpath=NULL, *outpath=NULL;

    for (int i=1;i<argc;i++){
        if (!strcmp(argv[i],"--base") && i+1<argc) base_win = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--hwin") && i+1<argc) hwin = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--k") && i+1<argc) k = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--post") && i+1<argc) post_win = atoi(argv[++i]);
        else if (!inpath) inpath = argv[i];
        else if (!outpath) outpath = argv[i];
        else usage(argv[0]);
    }
    if (!inpath || !outpath) usage(argv[0]);
    if (base_win<3 || (base_win&1)==0 || base_win>127){ fprintf(stderr,"--base must be odd 3..127\n"); return 1; }
    if (hwin<3 || (hwin&1)==0 || hwin>127){ fprintf(stderr,"--hwin must be odd 3..127\n"); return 1; }
    if (post_win<1 || (post_win%2)==0 || post_win>9){ fprintf(stderr,"--post must be odd 1..9\n"); return 1; }

    size_t tail=0, nspec = full_records(inpath, &tail);
    if (nspec==0){ fprintf(stderr,"No full records.\n"); return 1; }
    if (tail) fprintf(stderr,"Warning: ignoring %zu trailing bytes (partial record)\n", tail);

    FILE *fin=fopen(inpath,"rb"); if(!fin){ perror("fopen in"); return 1; }
    FILE *fout=fopen(outpath,"wb"); if(!fout){ perror("fopen out"); fclose(fin); return 1; }

    Record rec;
    float base[NCHAN], res[NCHAN];

    for (size_t s=0; s<nspec; ++s){
        must_read(&rec.ts,   sizeof(float), 1,             fin,  "ts");
        must_read(&rec.freq, sizeof(float), NCHAN,         fin,  "freq");
        must_read(&rec.spec, sizeof(float), NINP*NCHAN,    fin,  "spec");

        for (int inp=0; inp<NINP; ++inp){
            // Stage 1: baseline
            baseline_median(rec.spec[inp], NCHAN, base_win, base);
            // residual
            for (int kx=0;kx<NCHAN;++kx) res[kx] = rec.spec[inp][kx] - base[kx];
            // Stage 2: Hampel on residual
            hampel_residual(res, NCHAN, hwin, k);
            // reconstruct
            for (int kx=0;kx<NCHAN;++kx) rec.spec[inp][kx] = base[kx] + res[kx];
            // Stage 3: tiny post smooth
            post_median(rec.spec[inp], NCHAN, post_win);
        }

        must_write(&rec.ts,   sizeof(float), 1,             fout, "ts out");
        must_write(&rec.freq, sizeof(float), NCHAN,         fout, "freq out");
        must_write(&rec.spec, sizeof(float), NINP*NCHAN,    fout, "spec out");
    }

    fclose(fin); fclose(fout);
    fprintf(stderr,"Done: %s → %s (base=%d, hwin=%d, k=%.2f, post=%d)\n",
            inpath, outpath, base_win, hwin, k, post_win);
    return 0;
}
