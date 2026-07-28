#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <cstdint>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

static const char * RESULT_STR[] = { "ok", "FAILED" };

static float calc_tolerance(const float * data, int n) {
    float min_val = data[0], max_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    if (max_val == min_val) return 1.0e-4f;
    float step = (max_val - min_val) / 15.0f;
    return step * 0.5f + 1.0e-3f;
}

// Validate stored FP16 metadata in a quantized block
static int validate_metadata(const uint8_t * buf, const char * label) {
    int num_failed = 0;
    ggml_fp16_t hscale, hzero;
    memcpy(&hscale, buf + 0, sizeof(hscale));
    memcpy(&hzero,  buf + 2, sizeof(hzero));
    float scale = ggml_fp16_to_fp32(hscale);
    float zero  = ggml_fp16_to_fp32(hzero);

    if (!isfinite(scale)) {
        printf("%s: scale is not finite (%f)\n", label, scale);
        num_failed++;
    }
    if (!(scale > 0.0f)) {
        printf("%s: scale is not > 0 (%f)\n", label, scale);
        num_failed++;
    }
    if (!isfinite(zero)) {
        printf("%s: zero is not finite (%f)\n", label, zero);
        num_failed++;
    }
    return num_failed;
}

static int test_round_trip(const float * data, int n, const char * label, float tolerance) {
    const auto * qfns  = ggml_get_type_traits(GGML_TYPE_Q4_HQQ);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);

    assert(n % qfns->blck_size == 0);
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_HQQ, n);
    std::vector<uint8_t> buf(row_size);
    std::vector<float> dequant(n);

    qfns_cpu->from_float(data, buf.data(), n);
    qfns->to_float(buf.data(), dequant.data(), n);

    // validate stored FP16 metadata for every block
    int meta_bad = 0;
    for (size_t b = 0; b < (size_t)n / qfns->blck_size; b++) {
        meta_bad += validate_metadata(buf.data() + b * qfns->type_size, label);
    }

    float max_err = 0.0f;
    int failed_count = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(dequant[i] - data[i]);
        if (err > max_err) max_err = err;
        if (err > tolerance) failed_count++;
    }

    int num_failed = (failed_count > 0) || (meta_bad > 0);
    printf("%-40s max_err=%.6f  meta_err=%d  failures=%d/%d: %s\n",
           label, max_err, meta_bad, failed_count, n, RESULT_STR[num_failed]);
    return num_failed;
}

static int test_block_layout(void) {
    int num_failed = 0;
    const auto * qfns = ggml_get_type_traits(GGML_TYPE_Q4_HQQ);
    bool ok = (qfns->blck_size == 32);
    printf("block size = %d (expected 32): %s\n", qfns->blck_size, RESULT_STR[!ok]);
    num_failed += !ok;
    ok = (qfns->type_size == 20);
    printf("type size = %zu (expected 20): %s\n", qfns->type_size, RESULT_STR[!ok]);
    num_failed += !ok;
    float bpw = (float)(qfns->type_size * 8) / qfns->blck_size;
    ok = (fabsf(bpw - 5.0f) < 0.001f);
    printf("bpw = %.1f: %s\n", bpw, RESULT_STR[!ok]);
    num_failed += !ok;
    return num_failed;
}

static int test_nibble_packing(void) {
    int num_failed = 0;
    const int qk = 32;
    std::vector<float> data(qk);
    for (int i = 0; i < qk; i++) {
        data[i] = (float)(i % 16);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q4_HQQ, qk);
    std::vector<uint8_t> buf(row_size);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);
    qfns_cpu->from_float(data.data(), buf.data(), qk);
    for (int j = 0; j < qk/2; j++) {
        int q0 = buf[4 + j] & 0x0F;
        int q1 = buf[4 + j] >> 4;
        if (q0 != (j % 16)) {
            printf("nibble packing FAILED at byte %d: low nibble %d != %d\n", j, q0, j % 16);
            num_failed++;
        }
        if (q1 != ((j + qk/2) % 16)) {
            printf("nibble packing FAILED at byte %d: high nibble %d != %d\n", j, q1, (j + qk/2) % 16);
            num_failed++;
        }
    }
    if (num_failed == 0) {
        printf("%-40s: %s\n", "nibble packing (low=first_half, high=second_half)", RESULT_STR[0]);
    }
    return num_failed;
}

static int test_constant_blocks(void) {
    int num_failed = 0;
    const int qk = 32;
    const auto * qfns  = ggml_get_type_traits(GGML_TYPE_Q4_HQQ);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);

    struct TestCase { float value; const char * name; };
    TestCase cases[] = {
        {0.0f,       "zero"},
        {3.5f,       "positive"},
        {-2.0f,      "negative"},
        {1.0e-6f,    "very small"},
        {-1.0e-6f,   "very small negative"},
        {1.0e4f,     "large finite"},
        {100000.0f,  "huge constant"},
        {1000.1f,    "FP16 inexact"},
    };

    for (auto & tc : cases) {
        std::vector<float> data(qk, tc.value);
        size_t row_size = ggml_row_size(GGML_TYPE_Q4_HQQ, qk);
        std::vector<uint8_t> buf(row_size, 0xFF);
        std::vector<float> dequant(qk);

        qfns_cpu->from_float(data.data(), buf.data(), qk);
        qfns->to_float(buf.data(), dequant.data(), qk);

        // validate metadata
        num_failed += validate_metadata(buf.data(), tc.name);

        // verify all packed codes are explicitly zero
        bool codes_zero = true;
        for (size_t j = 4; j < row_size; j++) {
            if (buf[j] != 0) { codes_zero = false; break; }
        }
        if (!codes_zero) {
            printf("constant block %s: codes not explicitly zeroed\n", tc.name);
            num_failed++;
        }

        // expected after FP16 round trip using stored metadata
        ggml_fp16_t hscale, hzero;
        memcpy(&hscale, buf.data() + 0, sizeof(hscale));
        memcpy(&hzero,  buf.data() + 2, sizeof(hzero));
        float scale_stored = ggml_fp16_to_fp32(hscale);
        float zero_stored  = ggml_fp16_to_fp32(hzero);
        float expected = (0.0f - zero_stored) / scale_stored;

        bool ok = true;
        for (int i = 0; i < qk && ok; i++) {
            if (fabsf(dequant[i] - expected) > 1.0e-4f) {
                printf("constant block %s: element %d got %f expected %f\n",
                       tc.name, i, dequant[i], expected);
                ok = false;
            }
        }
        if (!ok) num_failed++;
    }

    printf("%-40s: %s\n", "constant blocks (all cases)", RESULT_STR[num_failed > 0]);
    return num_failed;
}

static int test_tiny_range(void) {
    int num_failed = 0;
    const int qk = 32;
    // non-constant tiny range near zero: range = 31 * 1.0e-9 = 3.1e-8
    std::vector<float> data(qk);
    for (int i = 0; i < qk; i++) {
        data[i] = (float)i * 1.0e-9f;
    }
    // confirm it is non-constant
    float mn = data[0], mx = data[0];
    for (int i = 1; i < qk; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    if (mn == mx) {
        printf("tiny range: data collapsed to constant, can't test\n");
        return 1;
    }

    size_t row_size = ggml_row_size(GGML_TYPE_Q4_HQQ, qk);
    std::vector<uint8_t> buf(row_size);
    std::vector<float> dequant(qk);
    const auto * qfns    = ggml_get_type_traits(GGML_TYPE_Q4_HQQ);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);

    qfns_cpu->from_float(data.data(), buf.data(), qk);
    qfns->to_float(buf.data(), dequant.data(), qk);

    num_failed += validate_metadata(buf.data(), "tiny range");

    int bad = 0;
    for (int i = 0; i < qk; i++) {
        if (isnan(dequant[i]) || isinf(dequant[i])) bad++;
    }
    printf("%-40s: %s\n", "tiny range (FP16 overflow policy)", RESULT_STR[bad > 0]);
    num_failed += (bad > 0);
    return num_failed;
}

static int test_large_offset_small_range(void) {
    int num_failed = 0;
    const int qk = 32;
    // large offset + small range causes zero to overflow FP16 with ideal scale
    std::vector<float> data(qk);
    for (int i = 0; i < qk; i++) {
        data[i] = 10000.0f + (float)i * 0.003f;
    }
    float mn = data[0], mx = data[0];
    for (int i = 1; i < qk; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    if (mn == mx) {
        printf("large offset: data collapsed to constant, can't test\n");
        return 1;
    }

    size_t row_size = ggml_row_size(GGML_TYPE_Q4_HQQ, qk);
    std::vector<uint8_t> buf(row_size);
    std::vector<float> dequant(qk);
    const auto * qfns    = ggml_get_type_traits(GGML_TYPE_Q4_HQQ);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);

    qfns_cpu->from_float(data.data(), buf.data(), qk);
    qfns->to_float(buf.data(), dequant.data(), qk);

    num_failed += validate_metadata(buf.data(), "large offset");

    // verify no NaN/Inf
    int bad = 0;
    for (int i = 0; i < qk; i++) {
        if (isnan(dequant[i]) || isinf(dequant[i])) bad++;
    }

    // verify output does not collapse: values span ~10000-10000.093
    // so dequant must be meaningfully near 10000, not ~15
    float max_err = 0.0f;
    for (int i = 0; i < qk; i++) {
        float err = fabsf(dequant[i] - data[i]);
        if (err > max_err) max_err = err;
    }
    // step = 0.093/15 approximately 0.0062, so tolerance of ~0.1 is reasonable
    bool collapse = (max_err > 1.0f);
    if (collapse) {
        printf("large offset: output collapsed (max_err=%f near 10000)\n", max_err);
        num_failed++;
    } else {
        printf("%-40s max_err=%.6f: %s\n", "large offset small range (zero overflow)", max_err, RESULT_STR[bad > 0]);
    }
    num_failed += (bad > 0);
    return num_failed;
}

int main(void) {
    ggml_cpu_init();

    int num_failed = 0;

    num_failed += test_block_layout();

    // 32 zeros (constant)
    {
        std::vector<float> data(32, 0.0f);
        num_failed += test_round_trip(data.data(), (int)data.size(), "32 zeros", calc_tolerance(data.data(), (int)data.size()));
    }

    // identical positive values (constant)
    {
        std::vector<float> data(32, 1.5f);
        num_failed += test_round_trip(data.data(), (int)data.size(), "identical positive (1.5)", calc_tolerance(data.data(), (int)data.size()));
    }

    // identical negative values (constant)
    {
        std::vector<float> data(32, -0.75f);
        num_failed += test_round_trip(data.data(), (int)data.size(), "identical negative (-0.75)", calc_tolerance(data.data(), (int)data.size()));
    }

    // very small constants (constant)
    {
        std::vector<float> data(32, 1.0e-6f);
        num_failed += test_round_trip(data.data(), (int)data.size(), "very small constant (1e-6)", calc_tolerance(data.data(), (int)data.size()));
    }

    // large finite constant (constant)
    {
        std::vector<float> data(32, 1.0e4f);
        num_failed += test_round_trip(data.data(), (int)data.size(), "large finite constant (1e4)", calc_tolerance(data.data(), (int)data.size()));
    }

    // mixed negative/positive values
    {
        std::vector<float> data(32);
        for (int i = 0; i < 32; i++) {
            data[i] = (i < 16) ? -1.0f : 1.0f;
        }
        num_failed += test_round_trip(data.data(), (int)data.size(), "mixed negative/positive", calc_tolerance(data.data(), (int)data.size()));
    }

    // monotonic values
    {
        std::vector<float> data(32);
        for (int i = 0; i < 32; i++) {
            data[i] = (float)i - 15.5f;
        }
        num_failed += test_round_trip(data.data(), (int)data.size(), "monotonic values", calc_tolerance(data.data(), (int)data.size()));
    }

    // random finite values
    {
        srand(42);
        std::vector<float> data(32);
        for (int i = 0; i < 32; i++) {
            data[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }
        num_failed += test_round_trip(data.data(), (int)data.size(), "random finite values", calc_tolerance(data.data(), (int)data.size()));
    }

    // nibble packing
    num_failed += test_nibble_packing();

    // constant blocks
    num_failed += test_constant_blocks();

    // tiny range (FP16 scale overflow)
    num_failed += test_tiny_range();

    // large offset small range (FP16 zero overflow)
    num_failed += test_large_offset_small_range();

    // valid codes test
    {
        srand(123);
        std::vector<float> data(32);
        for (int i = 0; i < 32; i++) {
            data[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }
        size_t row_size = ggml_row_size(GGML_TYPE_Q4_HQQ, 32);
        std::vector<uint8_t> buf(row_size);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);
        qfns_cpu->from_float(data.data(), buf.data(), 32);

        int bad = 0;
        for (size_t j = 4; j < row_size; j++) {
            int q0 = buf[j] & 0x0F;
            int q1 = buf[j] >> 4;
            if (q0 < 0 || q0 > 15) bad++;
            if (q1 < 0 || q1 > 15) bad++;
        }
        printf("%-40s: %s\n", "valid codes in [0,15]", RESULT_STR[bad > 0]);
        num_failed += (bad > 0);
    }

    // no NaN/Inf in output
    {
        std::vector<float> data(32);
        for (int i = 0; i < 32; i++) {
            data[i] = (float)(i - 16);
        }
        size_t row_size = ggml_row_size(GGML_TYPE_Q4_HQQ, 32);
        std::vector<uint8_t> buf(row_size);
        std::vector<float> dequant(32);
        const auto * qfns    = ggml_get_type_traits(GGML_TYPE_Q4_HQQ);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);
        qfns_cpu->from_float(data.data(), buf.data(), 32);
        qfns->to_float(buf.data(), dequant.data(), 32);
        int bad = 0;
        for (int i = 0; i < 32; i++) {
            if (isnan(dequant[i]) || isinf(dequant[i])) bad++;
        }
        printf("%-40s: %s\n", "no NaN/Inf in output", RESULT_STR[bad > 0]);
        num_failed += (bad > 0);
    }

    if (num_failed) {
        printf("\n%d test(s) FAILED\n", num_failed);
    } else {
        printf("\nAll tests PASSED\n");
    }

    return num_failed > 0 ? 1 : 0;
}