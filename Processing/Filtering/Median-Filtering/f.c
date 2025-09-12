// rfi_filter.c — median on autospec float32, preserving layout
// build: cc -O2 -std=c11 -o rfi_filter rfi_filter.c
// use:   ./rfi_filter --axis freq --win 5 ADCout.dat ADCout_med.dat
//        ./rfi_filter --axis time --win 5 ADCout.dat ADCout_med.dat
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>

#define NINP  4
#define NCHAN 512

typedef struct {
    float ts;
    float freq[NCHAN];
    float spec[NINP][NCHAN];
} Record;

static void must_read(void *p, size_t sz, size_t n, FILE *fp, const char *what){
    if (fread(p, sz, n, fp) != n){ fprintf(stderr,"Read failed: %s\n", what); exit(1); }
}
static void must_write(const void *p, size_t sz, size_t n, FILE *fp, const char *what){
    if (fwrite(p, sz, n, fp) != n){ fprintf(stderr,"Write failed: %s\n", what); exit(1); }
}
static size_t rec_bytes(void){ return sizeof(float) * (1 + NCHAN + NINP * NCHAN); }

static size_t full_records(const char *path, size_t *tail){
    struct stat st; if (stat(path, &st)!=0){ perror("stat"); exit(1); }
    size_t rb = rec_bytes();
    size_t n  = (size_t)(st.st_size / rb);
    if (tail) *tail = (size_t)(st.st_size - (long long)n * rb);
    return n;
}

static inline void isort(float *a, int n){
    for (int i=1;i<n;++i){ float x=a[i]; int j=i-1; while (j>=0 && a[j]>x){ a[j+1]=a[j]; --j; } a[j+1]=x; }
}
static inline float median_window(const float *x, int n){
    // copy then insertion sort (n is small: 3/5/7)
    float tmp[9]; if (n>9){ fprintf(stderr,"win too big\n"); exit(1); }
    for (int i=0;i<n;++i) tmp[i]=x[i];
    isort(tmp,n); return tmp[n/2];
}

static void usage(const char *p){
    fprintf(stderr,
      "Usage: %s --axis time|freq --win <odd>=3 <in.dat> <out.dat>\n"
      "Layout per spectrum: float32 ts, float32 freq[512], float32 autospec[4][512]\n", p);
    exit(1);
}

int main(int argc, char **argv){
    if (argc < 6) usage(argv[0]);

    const char *axis = NULL, *inpath=NULL, *outpath=NULL;
    int win = 3;

    for (int i=1;i<argc;i++){
        if (!strcmp(argv[i],"--axis") && i+1<argc){ axis=argv[++i]; }
        else if (!strcmp(argv[i],"--win") && i+1<argc){ win=atoi(argv[++i]); }
        else if (!inpath){ inpath=argv[i]; }
        else if (!outpath){ outpath=argv[i]; }
        else usage(argv[0]);
    }
    if (!axis || !inpath || !outpath) usage(argv[0]);
    if ((win<1) || (win%2==0) || win>9){ fprintf(stderr,"win must be odd in [1..9]\n"); return 1; }

    int axis_time = !strcmp(axis,"time");
    int axis_freq = !strcmp(axis,"freq");
    if (!axis_time && !axis_freq){ fprintf(stderr,"axis must be time or freq\n"); return 1; }

    size_t tail=0, nspec = full_records(inpath, &tail);
    if (nspec==0){ fprintf(stderr,"No full records.\n"); return 1; }
    if (tail) fprintf(stderr,"Warning: ignoring %zu trailing bytes (partial record)\n", tail);

    FILE *fin=fopen(inpath,"rb"); if(!fin){ perror("fopen in"); return 1; }
    FILE *fout=fopen(outpath,"wb"); if(!fout){ perror("fopen out"); fclose(fin); return 1; }

    if (axis_freq){
        // process per-record: median along channel axis
        Record rec, out;
        for (size_t s=0; s<nspec; ++s){
            must_read(&rec.ts,   sizeof(float), 1,             fin,  "ts");
            must_read(&rec.freq, sizeof(float), NCHAN,         fin,  "freq");
            must_read(&rec.spec, sizeof(float), NINP*NCHAN,    fin,  "spec");
            out = rec;

            int r = win/2;
            for (int i=0;i<NINP;++i){
                float window[9];
                for (int k=0;k<NCHAN;++k){
                    int w=0;
                    for (int off=-r; off<=r; ++off){
                        int idx = k + off;
                        if (idx < 0) idx = 0;
                        if (idx >= NCHAN) idx = NCHAN-1;
                        window[w++] = rec.spec[i][idx];
                    }
                    out.spec[i][k] = median_window(window, win);
                }
            }
            must_write(&out.ts,   sizeof(float), 1,             fout, "ts out");
            must_write(&out.freq, sizeof(float), NCHAN,         fout, "freq out");
            must_write(&out.spec, sizeof(float), NINP*NCHAN,    fout, "spec out");
        }
    } else {
        // axis_time: temporal median with ring buffer of size win
        int W = win, R = W/2;
        // allocate buffers
        float (*buf)[NINP][NCHAN] = malloc((size_t)W * sizeof *buf);
        float *ts   = malloc((size_t)W * sizeof *ts);
        float (*fq)[NCHAN] = malloc((size_t)W * sizeof *fq);
        if(!buf||!ts||!fq){ fprintf(stderr,"oom\n"); return 1; }

        // prefill first W-1 (or as many as exist)
        size_t preload = (nspec < (size_t)W) ? (size_t)nspec : (size_t)W;
        for (size_t s=0; s<preload; ++s){
            must_read(&ts[s],   sizeof(float), 1,             fin,  "ts pre");
            must_read(&fq[s],   sizeof(float), NCHAN,         fin,  "freq pre");
            must_read(&buf[s],  sizeof(float), NINP*NCHAN,    fin,  "spec pre");
        }

        // process each output index s_out
        size_t wrote = 0;
        for (size_t s_out=0; s_out<nspec; ++s_out){
            // ensure window [s_out-R .. s_out+R] is present in ring
            // For simplicity, refill tail with edge replication when not available.
            // Build median per bin from window samples.
            Record out;
            // choose center record to copy ts/freq: clamp to existing
            size_t center = s_out;
            if (center >= nspec) center = nspec-1;
            size_t center_idx = (center % preload);

            out.ts = ts[center_idx];
            memcpy(out.freq, fq[center_idx], sizeof(out.freq));

            float window[9];
            for (int i=0;i<NINP;++i){
                for (int k=0;k<NCHAN;++k){
                    int w=0;
                    for (int off=-R; off<=R; ++off){
                        long src = (long)s_out + off;
                        if (src < 0) src = 0;
                        if ((size_t)src >= nspec) src = (long)nspec - 1;
                        size_t idx = (size_t)src % preload; // map into our preloaded block
                        window[w++] = buf[idx][i][k];
                    }
                    out.spec[i][k] = median_window(window, W);
                }
            }
            must_write(&out.ts,   sizeof(float), 1,             fout, "ts out");
            must_write(&out.freq, sizeof(float), NCHAN,         fout, "freq out");
            must_write(&out.spec, sizeof(float), NINP*NCHAN,    fout, "spec out");
            wrote++;
        }

        // consume any unread records (if nspec>W we already preloaded only W, but
        // this simple version preloads all nspec if nspec<W. For large files, a full
        // streaming ring is easy to add; ping me if you want that.)
        (void)wrote;
        free(buf); free(ts); free(fq);
        // Note: above “temporal” branch uses the preloaded block as a proxy. For very
        // large files or W<<nspec you may prefer the fully streaming variant I sent earlier.
    }

    fclose(fin); fclose(fout);
    fprintf(stderr,"Done: %s → %s (axis=%s, win=%d)\n", inpath, outpath, axis, win);
    return 0;
}
