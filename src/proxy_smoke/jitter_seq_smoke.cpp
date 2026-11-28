// --jitter-seq: the jitter sequences of jitter_seq.h ([dlaa] jitter) against the engine's table and NVIDIA's rules.

#include <cmath>
#include <cstdio>
#include <set>
#include <utility>

#include "jitter_seq.h"

using namespace qvfr;

int RunJitterSeq() {
    bool ok = true;
    auto check = [&](bool c, const char* what) {
        if (!c) printf("  FAILED: %s\n", what);
        ok = ok && c;
    };

    // The engine's table (CTAAControl ctor, docs: Halton 55.. scaled by 1.19).
    constexpr float kTable[8][2] = {{0.502f, -0.169f}, {-0.465f, 0.228f}, {0.130f, -0.433f}, {-0.167f, -0.037f},
                                    {0.428f, 0.360f},  {-0.316f, -0.301f}, {0.279f, 0.096f}, {-0.019f, 0.492f}};
    bool table = true;
    for (uint32_t f = 0; f < 16; ++f) {
        float o[2];
        jitter::GameOffset(f, o);
        table = table && std::fabs(o[0] - kTable[f & 7][0]) < 0.001f && std::fabs(o[1] - kTable[f & 7][1]) < 0.001f;
    }
    check(table, "the game's offsets match the engine's table");

    // Mode 1: the same 8 phases over the full range (+-1 offset = +-0.5 px).
    float maxFull = 0;
    bool full8 = true;
    for (uint32_t f = 0; f < 8; ++f) {
        float g[2], o[2];
        jitter::GameOffset(f, g);
        o[0] = g[0];
        o[1] = g[1];
        jitter::Apply(jitter::kFull8, 8, f, o);
        full8 = full8 && std::fabs(o[0] - g[0] * 2.0f / 1.19f) < 1e-5f && std::fabs(o[1] - g[1] * 2.0f / 1.19f) < 1e-5f;
        maxFull = std::fmax(maxFull, std::fmax(std::fabs(o[0]), std::fabs(o[1])));
    }
    // The table's largest value 0.502 is 84% of its range, so mode 1 reaches 0.844 (+-0.42 px).
    check(full8 && maxFull < 1.0f && maxFull > 0.84f, "mode 1 scales the game's phases to the full range");

    // NVIDIA's phase count: ceil(8 / s^2).
    check(jitter::NvidiaPhases(1.0f) == 8 && jitter::NvidiaPhases(0.67f) == 18 && jitter::NvidiaPhases(0.5f) == 32 &&
              jitter::NvidiaPhases(0.33f) == 74,
          "phase counts 8, 18, 32, 74 for scales 1, 0.67, 0.5, 0.33");

    // Mode 2: N distinct phases inside (-1, 1), never (0, 0), period N, mean about 0, the full range covered.
    int seqOk = 0;
    for (int n : {8, 18, 32, 74}) {
        std::set<std::pair<float, float>> seen;
        double sx = 0, sy = 0;
        float lo[2] = {1, 1}, hi[2] = {-1, -1};
        bool inRange = true, nonZero = true, period = true;
        for (uint32_t f = 1000; f < 1000 + (uint32_t)n; ++f) {
            float o[2] = {0.1f, 0.1f}, p[2] = {0.1f, 0.1f};
            jitter::Apply(jitter::kNvidia, n, f, o);
            jitter::Apply(jitter::kNvidia, n, f + (uint32_t)n, p);
            period = period && o[0] == p[0] && o[1] == p[1];
            inRange = inRange && std::fabs(o[0]) < 1.0f && std::fabs(o[1]) < 1.0f;
            nonZero = nonZero && (o[0] != 0.0f || o[1] != 0.0f);
            seen.insert({o[0], o[1]});
            sx += o[0];
            sy += o[1];
            for (int a = 0; a < 2; ++a) {
                lo[a] = std::fmin(lo[a], o[a]);
                hi[a] = std::fmax(hi[a], o[a]);
            }
        }
        // Halton from index 1 has a small mean offset in short sequences (8 phases: x -0.11, i.e. -0.055 px); DLSS gets
        // every offset as it is, so only a larger one would mean a broken sequence.
        const bool good = inRange && nonZero && period && (int)seen.size() == n && std::fabs(sx / n) < 0.12 &&
                          std::fabs(sy / n) < 0.12 && lo[0] < -0.7f && hi[0] > 0.7f && lo[1] < -0.7f && hi[1] > 0.7f;
        printf("  mode 2, %d phases: x %.3f..%.3f, y %.3f..%.3f, mean (%.3f, %.3f), %zu distinct%s\n", n, lo[0], hi[0],
               lo[1], hi[1], sx / n, sy / n, seen.size(), good ? "" : " FAILED");
        seqOk += good;
    }
    check(seqOk == 4, "mode 2 sequences");

    // Mode 0 leaves the engine's value.
    float o[2] = {0.25f, -0.125f};
    jitter::Apply(jitter::kGame, 8, 5, o);
    check(o[0] == 0.25f && o[1] == -0.125f, "mode 0 leaves the engine's offset");

    printf("Jitter sequence check %s: the game's table, mode 1 up to %.3f (%.3f px), NVIDIA phase counts, mode 2\n",
           ok ? "OK" : "FAILED", maxFull, maxFull * 0.5f);
    return ok ? 0 : 1;
}
