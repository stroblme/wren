/* wren.c — dependency-free CPU inference for packed ternary Wren weights.
 *
 * Loads the format wren/export.py writes (2-bit attention kernels,
 * fp32 embedding and norms; a metadata trailer is not read), decodes
 * greedily with a KV cache and stops at a blank line. Pinned
 * byte-for-byte against wren/runtime.py's generate by tests/test_runtime.py.
 *
 *   cc -O3 -o wren-rt runtime/wren.c -lm
 *   ./wren-rt wren/wren.bin "$(printf 'kitchen_light=false\nturn on the kitchen light')"
 *   printf '<target lines>\n<command>\n' | ./wren-rt wren/wren.bin -n 96
 *
 * ponytail: single-threaded, kernels unpacked to scale-folded floats in
 * RAM (~8 MB); SIMD intrinsics and in-register 2-bit unpacking are the
 * upgrade path if this ever becomes the bottleneck.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPS 1e-6f
#define MAX_HEAD_DIM 256
#define MAX_CONTEXT 8192

typedef struct {
    float *norm, *q_norm, *k_norm; /* (dim), (head_dim), (head_dim) */
    float *wqkv, *wo;              /* (3*dim, dim), (dim, dim) row-major */
    float wqkv_scale, wo_scale;
    float *k_cache, *v_cache; /* (context, heads, head_dim) */
} Layer;

typedef struct {
    int vocab, dim, n_layers, n_heads, context, head_dim;
    float *embed, *final_norm; /* (vocab, dim), (dim) */
    Layer *layers;
} Model;

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    return p;
}

static float *read_f32(FILE *f, size_t n) {
    float *p = xmalloc(n * sizeof(float));
    if (fread(p, sizeof(float), n, f) != n) { fprintf(stderr, "truncated file\n"); exit(1); }
    return p;
}

/* Unpacked to float at load: the scale is folded in, so the hot matvec
 * is a plain float dot product the compiler vectorizes. ~8 MB of RAM
 * buys the SIMD path; in-register 2-bit unpacking is the upgrade. */
static float *read_packed(FILE *f, size_t n, float scale) {
    size_t bytes = (n + 3) / 4;
    uint8_t *raw = xmalloc(bytes);
    if (fread(raw, 1, bytes, f) != bytes) { fprintf(stderr, "truncated file\n"); exit(1); }
    float *q = xmalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        q[i] = ((int)((raw[i / 4] >> (2 * (i % 4))) & 3) - 1) * scale;
    free(raw);
    return q;
}

static Model load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char magic[4];
    int32_t hdr[6];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "WREN", 4) != 0 ||
        fread(hdr, sizeof(int32_t), 6, f) != 6 || hdr[0] != 1) {
        fprintf(stderr, "%s: not a wren v1 pack\n", path);
        exit(1);
    }
    Model m = {.vocab = hdr[1], .dim = hdr[2], .n_layers = hdr[3],
               .n_heads = hdr[4], .context = hdr[5]};
    m.head_dim = m.n_heads > 0 ? m.dim / m.n_heads : 0;
    if (m.head_dim < 2 || m.head_dim > MAX_HEAD_DIM || m.head_dim % 2 ||
        m.dim != m.head_dim * m.n_heads || m.vocab < 256 || m.n_layers < 1 ||
        m.context < 1 || m.context > MAX_CONTEXT) {
        fprintf(stderr, "%s: bad header\n", path);
        exit(1);
    }
    m.embed = read_f32(f, (size_t)m.vocab * m.dim);
    m.final_norm = read_f32(f, m.dim);
    m.layers = xmalloc(m.n_layers * sizeof(Layer));
    for (int l = 0; l < m.n_layers; l++) {
        Layer *ly = &m.layers[l];
        ly->norm = read_f32(f, m.dim);
        ly->q_norm = read_f32(f, m.head_dim);
        ly->k_norm = read_f32(f, m.head_dim);
        if (fread(&ly->wqkv_scale, sizeof(float), 1, f) != 1) { fprintf(stderr, "truncated file\n"); exit(1); }
        ly->wqkv = read_packed(f, (size_t)3 * m.dim * m.dim, ly->wqkv_scale);
        if (fread(&ly->wo_scale, sizeof(float), 1, f) != 1) { fprintf(stderr, "truncated file\n"); exit(1); }
        ly->wo = read_packed(f, (size_t)m.dim * m.dim, ly->wo_scale);
        size_t cache = (size_t)m.context * m.dim;
        ly->k_cache = xmalloc(cache * sizeof(float));
        ly->v_cache = xmalloc(cache * sizeof(float));
    }
    fclose(f);
    return m;
}

static void rmsnorm(const float *x, const float *scale, float *out, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + EPS);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * scale[i];
}

static void rope(float *x, int pos, int half) {
    for (int i = 0; i < half; i++) {
        float angle = pos / powf(10000.0f, (float)i / half);
        float c = cosf(angle), s = sinf(angle);
        float a = x[i], b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

/* y = W @ x, W (rows, cols) row-major with the ternary scale folded in. */
static void matvec(const float *w, const float *x, float *y, int rows,
                   int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row = w + (size_t)r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) acc += row[c] * x[c];
        y[r] = acc;
    }
}

/* One decode step: logits for `token` at `pos` (caller-provided buffers). */
static void step(Model *m, int token, int pos, float *logits, float *x,
                 float *h, float *qkv, float *o, float *scores) {
    memcpy(x, m->embed + (size_t)token * m->dim, m->dim * sizeof(float));
    int hd = m->head_dim, half = hd / 2;
    for (int l = 0; l < m->n_layers; l++) {
        Layer *ly = &m->layers[l];
        rmsnorm(x, ly->norm, h, m->dim);
        matvec(ly->wqkv, h, qkv, 3 * m->dim, m->dim);
        for (int head = 0; head < m->n_heads; head++) {
            float *q = qkv + head * hd;
            float *k = qkv + m->dim + head * hd;
            float *v = qkv + 2 * m->dim + head * hd;
            float qn[MAX_HEAD_DIM], kn[MAX_HEAD_DIM];
            rmsnorm(q, ly->q_norm, qn, hd);
            rmsnorm(k, ly->k_norm, kn, hd);
            rope(qn, pos, half);
            rope(kn, pos, half);
            memcpy(ly->k_cache + ((size_t)pos * m->n_heads + head) * hd, kn,
                   hd * sizeof(float));
            memcpy(ly->v_cache + ((size_t)pos * m->n_heads + head) * hd, v,
                   hd * sizeof(float));
            float max = -1e30f;
            for (int j = 0; j <= pos; j++) {
                const float *kj = ly->k_cache + ((size_t)j * m->n_heads + head) * hd;
                float acc = 0.0f;
                for (int i = 0; i < hd; i++) acc += qn[i] * kj[i];
                scores[j] = acc / sqrtf((float)hd);
                if (scores[j] > max) max = scores[j];
            }
            float sum = 0.0f;
            for (int j = 0; j <= pos; j++) {
                scores[j] = expf(scores[j] - max);
                sum += scores[j];
            }
            float *oh = o + head * hd;
            memset(oh, 0, hd * sizeof(float));
            for (int j = 0; j <= pos; j++) {
                const float *vj = ly->v_cache + ((size_t)j * m->n_heads + head) * hd;
                float w = scores[j] / sum;
                for (int i = 0; i < hd; i++) oh[i] += w * vj[i];
            }
        }
        matvec(ly->wo, o, h, m->dim, m->dim);
        for (int i = 0; i < m->dim; i++) x[i] += h[i];
    }
    rmsnorm(x, m->final_norm, h, m->dim);
    for (int v = 0; v < m->vocab; v++) {
        const float *e = m->embed + (size_t)v * m->dim;
        float acc = 0.0f;
        for (int i = 0; i < m->dim; i++) acc += e[i] * h[i];
        logits[v] = acc;
    }
}

int main(int argc, char **argv) {
    const char *path = NULL, *prompt_arg = NULL;
    int n_new = 96;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            n_new = atoi(argv[++i]);
        else if (!path)
            path = argv[i];
        else
            prompt_arg = argv[i];
    }
    if (!path) {
        fprintf(stderr, "usage: %s weights.bin [-n bytes] [prompt]\n", argv[0]);
        return 2;
    }
    Model m = load_model(path);
    if (n_new < 1 || n_new >= m.context) {
        fprintf(stderr, "-n must be in 1..%d\n", m.context - 1);
        return 2;
    }

    /* Only the tail of an over-long prompt fits the context window: keep
     * the last `budget` bytes in a ring buffer. */
    size_t budget = (size_t)(m.context - n_new), total = 0;
    unsigned char *ring = xmalloc(budget);
    if (prompt_arg) {
        size_t n = strlen(prompt_arg);
        for (size_t i = 0; i < n; i++) ring[total++ % budget] = prompt_arg[i];
        if (n == 0 || prompt_arg[n - 1] != '\n') ring[total++ % budget] = '\n';
    } else {
        for (int c; (c = getchar()) != EOF;) ring[total++ % budget] = c;
    }
    size_t len = total < budget ? total : budget;
    if (len == 0) { fprintf(stderr, "empty prompt\n"); return 2; }

    float *logits = xmalloc(m.vocab * sizeof(float));
    float *x = xmalloc(m.dim * sizeof(float));
    float *h = xmalloc(3 * m.dim * sizeof(float));
    float *qkv = xmalloc(3 * m.dim * sizeof(float));
    float *o = xmalloc(m.dim * sizeof(float));
    float *scores = xmalloc(m.context * sizeof(float));

    int tokens[MAX_CONTEXT];
    int n = 0;
    for (size_t i = total - len; i < total; i++) tokens[n++] = ring[i % budget];
    int limit = (int)len + n_new;
    if (limit > m.context) limit = m.context;
    for (int pos = 0; pos < n; pos++) {
        step(&m, tokens[pos], pos, logits, x, h, qkv, o, scores);
        if (pos == n - 1 && n < limit) {
            int best = 0;
            for (int v = 1; v < m.vocab; v++)
                if (logits[v] > logits[best]) best = v;
            tokens[n++] = best;
            putchar(best);
            fflush(stdout);
            if (n >= 2 && tokens[n - 1] == '\n' && tokens[n - 2] == '\n') break;
        }
    }
    return 0;
}
