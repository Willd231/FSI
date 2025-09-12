#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <string.h>
#include <inttypes.h>  // for PRIu16

uint16_t *get_data(const char *filename, size_t *length) {
    struct stat st;
    if (stat(filename, &st) != 0) {
        perror("stat failed");
        return NULL;
    }

    if (st.st_size % sizeof(uint16_t) != 0) {
        fprintf(stderr, "file size not multiple of 2 bytes\n");
        return NULL;
    }

    *length = st.st_size / sizeof(uint16_t);
    uint16_t *data = malloc(*length * sizeof(uint16_t));
    if (!data) {
        perror("malloc failed");
        return NULL;
    }

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen failed");
        free(data);
        return NULL;
    }

    size_t n = fread(data, sizeof(uint16_t), *length, fp);
    if (n != *length) {
        perror("fread failed");
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);

    return data;
}

uint16_t median(uint16_t a, uint16_t b, uint16_t c) {
    if (a > b) { uint16_t t = a; a = b; b = t; }
    if (b > c) { uint16_t t = b; b = c; c = t; }
    if (a > b) { uint16_t t = a; a = b; b = t; }
    return b;
}

uint16_t *DoFilter(const uint16_t *data, size_t length) {
    uint16_t *out = malloc(length * sizeof(uint16_t));
    if (!out) {
        perror("malloc failed");
        return NULL;
    }

    if (length == 0) return out;
    if (length == 1) { out[0] = data[0]; return out; }

    out[0] = data[0];
    for (size_t i = 1; i + 1 < length; i++) {
        out[i] = median(data[i-1], data[i], data[i+1]);
    }
    out[length-1] = data[length-1];
    return out;
}

int writeData(const uint16_t *data, const char *outFilename, size_t length) {
    FILE *fp = fopen(outFilename, "wb");
    if (!fp) {
        perror("fopen failed");
        return -1;
    }

    if (fwrite(data, sizeof(uint16_t), length, fp) != length) {
        perror("fwrite failed");
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input file> <output file>\n", argv[0]);
        return 1;
    }

    size_t len = 0;
    uint16_t *data = get_data(argv[1], &len);
    if (!data) return 1;

    uint16_t *filtered = DoFilter(data, len);
    if (!filtered) { free(data); return 1; }

    if (writeData(filtered, argv[2], len) != 0) {
        free(data);
        free(filtered);
        return 1;
    }

    size_t preview = len < 10 ? len : 10;
    printf("First %zu values:\n", preview);
    for (size_t i = 0; i < preview; i++) {
        printf("%" PRIu16 " ", filtered[i]);
    }
    printf("\n");

    free(data);
    free(filtered);
    return 0;
}
