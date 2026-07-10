// archetypes.cu — ground-truth calibration kernels for the GPU-DAMOV classifier.
//
// DAMOV validated its six CPU classes on 100 held-out functions with frozen
// thresholds (97% correct, paper §3.5). The GPU analog needs kernels whose
// bottleneck class is known BY CONSTRUCTION — then the classifier must recover
// them blind. Each kernel below is designed to sit squarely in one G-class:
//
//   g1_triad    streaming a[i]+s*b[i] over huge arrays  -> G1 bandwidth-bound
//   g2_gather   random-index gather over a huge array   -> G2 coalescing-bound
//   g3_l2       repeated sweeps of an L2-resident array -> G3 L2-reuse-bound
//   g4_chase    dependent pointer-chase, 2 blocks       -> G4 latency-bound
//   g5_fma      register-resident FMA polynomial        -> G5 compute-bound
//   g6_shared   shared-memory ping-pong, no DRAM        -> G6 on-chip-bound
//   g7_dep      serial FMA chain at 1 warp/SM           -> G7 dependency-bound
//   g0_tiny     one warp, one add                       -> G0 no-signal
//
// The binary runs ONE archetype (argv[1]) for R launches (argv[2], default 8),
// prints per-launch kernel milliseconds (cudaEvent-timed, for the clock-domain
// intervention sweep) and a checksum (the QoR the harness verifies).
//
// Build:  nvcc -O3 -arch=native -o archetypes archetypes.cu
// Run:    ./archetypes g1_triad [launches]
//
// Used via the harness command adapter (configs/calibration/*.toml) so the
// SAME pipeline that classifies cuVSLAM classifies these — no special path.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

// ── G1: DRAM-bandwidth-bound — coalesced streaming triad, zero reuse ─────────
__global__ void g1_triad(const float* __restrict__ a, const float* __restrict__ b,
                         float* __restrict__ c, size_t n, float s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) c[i] = a[i] + s * b[i];
}

// ── G2: coalescing-bound — random gather (32 lanes hit 32 far-apart lines) ──
__global__ void g2_gather(const float* __restrict__ a, const unsigned* __restrict__ idx,
                          float* __restrict__ c, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n; i += stride) acc += a[idx[i]];
    if (acc == 12345.678f) c[0] = acc;    // keep the loads alive, ~no stores
}

// ── G3: L2-reuse-bound — hammer an array that FITS in L2 (reuse across sweeps)
__global__ void g3_l2(const float* __restrict__ a, float* __restrict__ c,
                      size_t n, int sweeps) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (int s = 0; s < sweeps; ++s)
        for (size_t j = i; j < n; j += stride) acc += a[j];
    if (acc == 12345.678f) c[0] = acc;
}

// ── G4: latency-bound — dependent pointer-chase, far too few warps to hide it.
// All 32 lanes of a warp follow the SAME pointer (a broadcast load: 1 sector
// per request, fully coalesced) so the kernel is pure DRAM *latency* with no
// scatter signature — otherwise the G2 rule fires first, and correctly so:
// a per-lane random chase is BOTH scattered and latency-bound.
__global__ void g4_chase(const unsigned* __restrict__ next, unsigned* __restrict__ out,
                         size_t hops) {
    unsigned p = blockIdx.x * 1024u + 1u;            // per-warp chain start
    for (size_t h = 0; h < hops; ++h) p = next[p];   // each load depends on the last
    out[blockIdx.x * blockDim.x + threadIdx.x] = p;
}

// ── G5: compute-bound — register-resident polynomial, full occupancy ────────
__global__ void g5_fma(float* __restrict__ c, size_t n, int iters) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = 1.0f + (float)(i & 1023) * 1e-6f;
    for (int k = 0; k < iters; ++k) {
        x = fmaf(x, 1.000001f, 1e-7f);
        x = fmaf(x, 0.999999f, 1e-7f);
    }
    c[i] = x;
}

// ── G6: on-chip-bound — bank-conflicted shared-memory hammering, NO barriers
// (a per-iteration __syncthreads makes `barrier` the dominant stall and the
// classifier rightly refuses G6 — the on-chip archetype must be limited by the
// shared-memory pipes themselves: short_scoreboard / mio_throttle dominant).
__global__ void g6_shared(float* __restrict__ c, int iters) {
    __shared__ float sm[1024];
    int t = threadIdx.x;
    sm[t] = (float)t;
    __syncthreads();                                  // one-time init only
    float acc = 0.f;
    int i0 = (t * 33) & 1023, i1 = (t * 17 + 1) & 1023;
    for (int k = 0; k < iters; ++k) {
        acc += sm[(i0 + k * 33) & 1023];              // stride-33 -> bank conflicts
        sm[(i1 + k * 17) & 1023] = acc;               // dependent LDS/STS chain
    }
    if (acc == 12345.678f) c[blockIdx.x] = acc;
}

// ── G7: dependency-bound — one warp per SM, serial FMA chain, memory idle ───
__global__ void g7_dep(float* __restrict__ c, int iters) {
    float x = 1.0f + (float)threadIdx.x * 1e-6f;
    for (int k = 0; k < iters; ++k)
        x = fmaf(x, 1.0000001f, 1e-9f);   // each FMA depends on the previous
    c[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ── G0: no-signal — a trivial launch-tax kernel ─────────────────────────────
__global__ void g0_tiny(float* __restrict__ c) {
    if (threadIdx.x == 0) c[0] += 1.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <archetype> [launches]\n", argv[0]); return 2; }
    const char* which = argv[1];
    int launches = argc > 2 ? atoi(argv[2]) : 8;

    int dev = 0; cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, dev));
    const int sms = prop.multiProcessorCount;

    // Sizes: big streams ≫ L2 (so G1/G2 defeat caches); g3 array fits in L2/3.
    const size_t N_BIG  = 96ull << 20;                       // 96M floats = 384 MB
    const size_t N_L2   = (size_t)(prop.l2CacheSize / 3) / sizeof(float);
    const size_t N_CHASE = 64ull << 20;                      // 64M nodes = 256 MB

    float *a = nullptr, *b = nullptr, *c = nullptr;
    unsigned *idx = nullptr, *nxt = nullptr, *uout = nullptr;
    CHECK(cudaMalloc(&c, (N_BIG + 2) * sizeof(float)));
    CHECK(cudaMemset(c, 0, (N_BIG + 2) * sizeof(float)));

    if (!strcmp(which, "g1_triad") || !strcmp(which, "g3_l2")) {
        size_t n = !strcmp(which, "g1_triad") ? N_BIG : N_L2;
        CHECK(cudaMalloc(&a, n * sizeof(float)));
        CHECK(cudaMemset(a, 0x3f, n * sizeof(float)));
        if (!strcmp(which, "g1_triad")) {
            CHECK(cudaMalloc(&b, n * sizeof(float)));
            CHECK(cudaMemset(b, 0x3f, n * sizeof(float)));
        }
    } else if (!strcmp(which, "g2_gather")) {
        CHECK(cudaMalloc(&a, N_BIG * sizeof(float)));
        CHECK(cudaMemset(a, 0x3f, N_BIG * sizeof(float)));
        CHECK(cudaMalloc(&idx, N_BIG * sizeof(unsigned)));
        // host-side LCG permutation-ish random indices (scatter across 384 MB)
        unsigned* h = (unsigned*)malloc(N_BIG * sizeof(unsigned));
        unsigned s = 12345u;
        for (size_t i = 0; i < N_BIG; ++i) { s = s * 1664525u + 1013904223u; h[i] = s % (unsigned)N_BIG; }
        CHECK(cudaMemcpy(idx, h, N_BIG * sizeof(unsigned), cudaMemcpyHostToDevice));
        free(h);
    } else if (!strcmp(which, "g4_chase")) {
        CHECK(cudaMalloc(&nxt, N_CHASE * sizeof(unsigned)));
        CHECK(cudaMalloc(&uout, 4096 * sizeof(unsigned)));
        unsigned* h = (unsigned*)malloc(N_CHASE * sizeof(unsigned));
        unsigned s = 99991u;                                  // random ring: defeat caches+prefetch
        for (size_t i = 0; i < N_CHASE; ++i) { s = s * 1664525u + 1013904223u; h[i] = s % (unsigned)N_CHASE; }
        CHECK(cudaMemcpy(nxt, h, N_CHASE * sizeof(unsigned), cudaMemcpyHostToDevice));
        free(h);
    }

    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    double total_ms = 0.0;
    for (int r = 0; r < launches; ++r) {
        CHECK(cudaEventRecord(t0));
        if (!strcmp(which, "g1_triad")) {
            g1_triad<<<sms * 8, 256>>>(a, b, c, N_BIG, 1.5f);
        } else if (!strcmp(which, "g2_gather")) {
            g2_gather<<<sms * 8, 256>>>(a, idx, c, N_BIG);
        } else if (!strcmp(which, "g3_l2")) {
            g3_l2<<<sms * 8, 256>>>(a, c, N_L2, 64);
        } else if (!strcmp(which, "g4_chase")) {
            g4_chase<<<2, 64>>>(nxt, uout, 200000);           // 128 threads total: occupancy ~0
        } else if (!strcmp(which, "g5_fma")) {
            g5_fma<<<sms * 8, 256>>>(c, (size_t)sms * 8 * 256, 40000);
        } else if (!strcmp(which, "g6_shared")) {
            g6_shared<<<sms * 4, 1024>>>(c, 20000);
        } else if (!strcmp(which, "g7_dep")) {
            g7_dep<<<sms, 32>>>(c, 2000000);                  // 1 warp/SM, pure dep chain
        } else if (!strcmp(which, "g0_tiny")) {
            g0_tiny<<<1, 32>>>(c);
        } else { fprintf(stderr, "unknown archetype %s\n", which); return 2; }
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, t0, t1));
        printf("launch=%d kernel_ms=%.3f\n", r, ms);
        total_ms += ms;
    }
    CHECK(cudaDeviceSynchronize());

    float sum = 0.f;
    CHECK(cudaMemcpy(&sum, c, sizeof(float), cudaMemcpyDeviceToHost));
    printf("mean_kernel_ms=%.3f\n", total_ms / launches);
    printf("checksum=%.6e\n", (double)sum + total_ms * 0.0);  // stable per archetype
    return 0;
}
