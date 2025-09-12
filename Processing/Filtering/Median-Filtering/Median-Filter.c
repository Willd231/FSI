// median.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <string.h>
#include <inttypes.h>
#include <errno.h>

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

static uint16_t* read_u16_le(const char *filename, size_t *length) {
    struct stat st;
    if (stat(filename, &st) != 0) die("stat");
    if (st.st_size % sizeof(uint16_t) != 0) {
        fprintf(stderr, "file size not multiple of 2 bytes\n");
        exit(1);
    }
    *length = (size_t)(st.st_size / sizeof(uint16_t));
    uint16_t *buf = (uint16_t*)malloc(*length * sizeof(uint16_t));
    if (!buf) die("malloc");

    FILE *fp = fopen(filename, "rb");
    if (!fp) die("fopen");

    size_t n = fread(buf, sizeof(uint16_t), *length, fp);
    if (n != *length) die("fread");
    fclose(fp);

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    for (size_t i = 0; i < *length; ++i) {
        uint16_t v = buf[i];
        buf[i] = (uint16_t)((v >> 8) | (v << 8));
    }
#endif
    return buf;
}

static void write_u16_le(const char *filename, const uint16_t *data, size_t length) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) die("fopen");
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // write as little-endian regardless of host
    for (size_t i = 0; i < length; ++i) {
        uint16_t v = data[i];
        uint16_t le = (uint16_t)((v >> 8) | (v << 8));
        if (fwrite(&le, sizeof(uint16_t), 1, fp) != 1) die("fwrite");
    }
#else
    if (fwrite(data, sizeof(uint16_t), length, fp) != length) die("fwrite");
#endif
    fclose(fp);
}

// insertion sort is fast for small windows
static void isort_u16(uint16_t *a, int n) {
    for (int i = 1; i < n; ++i) {
        uint16_t key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) { a[j+1] = a[j]; --j; }
        a[j+1] = key;
    }
}

static void median_filter(const uint16_t *in, uint16_t *out, size_t len, int win) {
    if (len == 0) return;
    if (win < 1 || (win & 1) == 0) {
        fprintf(stderr, "window size must be odd and >= 1\n");
        exit(1);
    }
    if (win == 1) { memcpy(out, in, len * sizeof(uint16_t)); return; }

    int r = win / 2;
    uint16_t *tmp = (uint16_t*)malloc((size_t)win * sizeof(uint16_t));
    if (!tmp) die("malloc");

    // edge replicate: clamp indices to [0, len-1]
    for (size_t i = 0; i < len; ++i) {
        int k = 0;
        for (int off = -r; off <= r; ++off) {
            long idx = (long)i + off;
            if (idx < 0) idx = 0;
            if ((size_t)idx >= len) idx = (long)len - 1;
            tmp[k++] = in[(size_t)idx];
        }
        isort_u16(tmp, win);
        out[i] = tmp[r];
    }
    free(tmp);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <input_u16_le.bin> <output_u16_le.bin> [window_odd>=3]\n", argv[0]);
        return 1;
    }
    int win = 3;
    if (argc == 4) {
        win = atoi(argv[3]);
        if (win < 3 || (win & 1) == 0) {
            fprintf(stderr, "window must be odd and >=3 (got %d)\n", win);
            return 1;
        }
    }

    size_t n = 0;
    uint16_t *x = read_u16_le(argv[1], &n);
    uint16_t *y = (uint16_t*)malloc(n * sizeof(uint16_t));
    if (!y) die("malloc");

    median_filter(x, y, n, win);
    write_u16_le(argv[2], y, n);

    // tiny preview
    size_t preview = n < 10 ? n : 10;
    fprintf(stdout, "len=%zu, window=%d\nfirst %zu values: ", n, win, preview);
    for (size_t i = 0; i < preview; ++i) fprintf(stdout, "%" PRIu16 " ", y[i]);
    fprintf(stdout, "\n");

    free(x);
    free(y);
    return 0;
}
