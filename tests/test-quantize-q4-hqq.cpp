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

    // CPU trait assertions for Q4_HQQ
    {
        const auto * cpu_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);
        bool ok = true;
        if (cpu_traits->from_float == nullptr) {
            printf("CPU trait from_float is null: FAILED\n");
            ok = false;
        }
        if (cpu_traits->vec_dot == nullptr) {
            printf("CPU trait vec_dot is null: FAILED\n");
            ok = false;
        }
        if (cpu_traits->vec_dot_type != GGML_TYPE_Q8_0) {
            printf("CPU trait vec_dot_type != Q8_0: FAILED\n");
            ok = false;
        }
        if (cpu_traits->nrows != 1) {
            printf("CPU trait nrows != 1: FAILED\n");
            ok = false;
        }
        printf("%-40s: %s\n", "CPU traits (from_float, vec_dot, vec_dot_type, nrows)", RESULT_STR[!ok]);
        num_failed += !ok;
    }

    //
    // Dot product tests
    //

    // Quantize to Q4_HQQ and Q8_0 then compute dot product via vec_dot
    // Reference: dequantize both sides and compute F32 dot product
    auto dot_q4_hqq_vs_f32 = [&](const float * data_q4, const float * data_q8, int n, const char * label) -> int {
        const auto * qfns    = ggml_get_type_traits(GGML_TYPE_Q4_HQQ);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);
        assert(n % qfns->blck_size == 0);

        size_t row_size_q4 = ggml_row_size(GGML_TYPE_Q4_HQQ, n);
        std::vector<uint8_t> buf_q4(row_size_q4);
        qfns_cpu->from_float(data_q4, buf_q4.data(), n);

        size_t row_size_q8 = ggml_row_size(GGML_TYPE_Q8_0, n);
        std::vector<uint8_t> buf_q8(row_size_q8);
        const auto * q8_traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
        q8_traits->from_float_ref(data_q8, buf_q8.data(), n);

        // Dequantize both sides for the reference dot product
        std::vector<float> deq_q4(n), deq_q8(n);
        qfns->to_float(buf_q4.data(), deq_q4.data(), n);
        q8_traits->to_float(buf_q8.data(), deq_q8.data(), n);

        float ref = 0.0f;
        for (int i = 0; i < n; i++) ref += deq_q4[i] * deq_q8[i];

        float dot_via_vec = 0.0f;
        const auto * cpu_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);
        if (cpu_traits->vec_dot) {
            cpu_traits->vec_dot(n, &dot_via_vec, 0, buf_q4.data(), 0, buf_q8.data(), 0, 1);
        } else {
            printf("vec_dot not registered for Q4_HQQ\n");
            return 1;
        }

        float err = fabsf(dot_via_vec - ref);
        float tol = fmaxf(1.0e-4f, fabsf(ref) * 1.0e-4f);
        bool ok = (err <= tol);
        if (!ok) {
            printf("%-40s vec_dot=%f ref=%f err=%f tol=%f: FAILED\n", label, dot_via_vec, ref, err, tol);
        } else {
            printf("%-40s vec_dot=%f ref=%f err=%f tol=%f: ok\n", label, dot_via_vec, ref, err, tol);
        }
        return ok ? 0 : 1;
    };

    // Random vectors
    {
        const int n = 32;
        srand(42);
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; i++) {
            a[i] = (float)(rand() % 20000 - 10000) / 100.0f;
            b[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }
        num_failed += dot_q4_hqq_vs_f32(a.data(), b.data(), n, "dot random vectors");
    }

    // Positive-only values
    {
        const int n = 32;
        srand(42);
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; i++) {
            a[i] = (float)(rand() % 1000) / 100.0f + 0.01f;
            b[i] = (float)(rand() % 1000) / 100.0f + 0.01f;
        }
        num_failed += dot_q4_hqq_vs_f32(a.data(), b.data(), n, "dot positive-only");
    }

    // Negative-only values
    {
        const int n = 32;
        srand(42);
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; i++) {
            a[i] = -((float)(rand() % 1000) / 100.0f + 0.01f);
            b[i] = -((float)(rand() % 1000) / 100.0f + 0.01f);
        }
        num_failed += dot_q4_hqq_vs_f32(a.data(), b.data(), n, "dot negative-only");
    }

    // Mixed-sign values
    {
        const int n = 32;
        srand(42);
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; i++) {
            a[i] = (float)(rand() % 20000 - 10000) / 100.0f;
            b[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }
        // Ensure mix of signs
        a[0] = -50.0f; b[0] = 30.0f;
        a[1] = 60.0f;  b[1] = -40.0f;
        num_failed += dot_q4_hqq_vs_f32(a.data(), b.data(), n, "dot mixed-sign");
    }

    // Constant Q4_HQQ blocks (all same value)
    {
        const int n = 32;
        std::vector<float> a(n, 1.5f);
        std::vector<float> b(n);
        srand(42);
        for (int i = 0; i < n; i++) {
            b[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }
        num_failed += dot_q4_hqq_vs_f32(a.data(), b.data(), n, "dot constant Q4 block");
    }

    // Q8 vectors with non-zero sum (tests zero-point correction)
    {
        const int n = 32;
        std::vector<float> a(n);
        srand(42);
        for (int i = 0; i < n; i++) {
            a[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }
        std::vector<float> b(n, 100.0f); // large positive constant -> non-zero sum
        num_failed += dot_q4_hqq_vs_f32(a.data(), b.data(), n, "dot Q8 non-zero sum");
    }

    // Multiple blocks (64 elements = 2 blocks)
    {
        const int n = 64;
        srand(42);
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; i++) {
            a[i] = (float)(rand() % 20000 - 10000) / 100.0f;
            b[i] = (float)(rand() % 20000 - 10000) / 100.0f;
        }
        num_failed += dot_q4_hqq_vs_f32(a.data(), b.data(), n, "dot 2 blocks (64 elts)");
    }

    // Fractional-zero regression test: guarantees fractional FP16 zero point
    // Values span min=-0.1, max=0.9 so zero = -min * scale = 0.1*15/1.0 = 1.5 exactly
    {
        const int n = 32;
        std::vector<float> a(n);
        for (int i = 0; i < n; i++) {
            a[i] = -0.1f + (float)i / 31.0f;
        }
        std::vector<float> b(n, 1.0f); // definitely non-zero sum (sum = 32)

        const auto * qfns    = ggml_get_type_traits(GGML_TYPE_Q4_HQQ);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);
        size_t row_size_q4 = ggml_row_size(GGML_TYPE_Q4_HQQ, n);
        std::vector<uint8_t> buf_q4(row_size_q4);
        qfns_cpu->from_float(a.data(), buf_q4.data(), n);

        // Read stored FP16 zero and assert it has a meaningful fractional part
        ggml_fp16_t hzero;
        memcpy(&hzero, buf_q4.data() + 2, sizeof(hzero));
        float zero_stored = ggml_fp16_to_fp32(hzero);
        float zero_frac = zero_stored - floorf(zero_stored);
        bool has_frac = (zero_frac > 0.001f) && (zero_frac < 0.999f);
        if (!has_frac) {
            printf("fractional-zero: stored zero %f has no meaningful fractional part (frac=%f)\n", zero_stored, zero_frac);
            num_failed++;
        }

        // Q8_0 quantize b
        size_t row_size_q8 = ggml_row_size(GGML_TYPE_Q8_0, n);
        std::vector<uint8_t> buf_q8(row_size_q8);
        const auto * q8_traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
        q8_traits->from_float_ref(b.data(), buf_q8.data(), n);

        // Dequantized reference (correct, float zero)
        std::vector<float> deq_q4(n), deq_q8(n);
        qfns->to_float(buf_q4.data(), deq_q4.data(), n);
        q8_traits->to_float(buf_q8.data(), deq_q8.data(), n);
        float ref_correct = 0.0f;
        for (int i = 0; i < n; i++) ref_correct += deq_q4[i] * deq_q8[i];

        // Intentionally broken reference using (int)zero
        float ref_broken = 0.0f;
        {
            // Reconstruct using integer-zero dequant
            ggml_fp16_t hscale;
            memcpy(&hscale, buf_q4.data(), sizeof(hscale));
            float scale_stored = ggml_fp16_to_fp32(hscale);
            int zero_int = (int)zero_stored;
            for (int i = 0; i < n; i++) {
                float w = 0.0f;
                int block_idx = i / 32;
                int elem = i % 32;
                int byte_idx = elem % (int)(qfns->blck_size / 2);
                int nibble = (elem < (int)(qfns->blck_size / 2)) ? (buf_q4[block_idx * 20 + 4 + byte_idx] & 0x0F)
                                                   : (buf_q4[block_idx * 20 + 4 + byte_idx] >> 4);
                w = ((float)nibble - (float)zero_int) / scale_stored;
                ref_broken += w * deq_q8[i];
            }
        }

        float dot_via_vec = 0.0f;
        qfns_cpu->vec_dot(n, &dot_via_vec, 0, buf_q4.data(), 0, buf_q8.data(), 0, 1);

        float err_correct = fabsf(dot_via_vec - ref_correct);
        float err_broken   = fabsf(dot_via_vec - ref_broken);
        float tol = fmaxf(1.0e-4f, fabsf(ref_correct) * 1.0e-4f);

        bool match_correct = (err_correct <= tol);
        bool differs_from_broken = (err_broken > tol * 10.0f);

        if (!match_correct) {
            printf("fractional-zero: vec_dot=%f ref=%f err=%f tol=%f: FAILED (does not match correct ref)\n",
                   dot_via_vec, ref_correct, err_correct, tol);
            num_failed++;
        }
        if (!differs_from_broken) {
            printf("fractional-zero: vec_dot=%f broken_ref=%f err=%f: FAILED (does not differ from int-zero ref)\n",
                   dot_via_vec, ref_broken, err_broken);
            num_failed++;
        }
        if (match_correct && differs_from_broken) {
            printf("%-40s vec_dot=%f correct=%f broken=%f: ok\n", "dot fractional-zero regression",
                   dot_via_vec, ref_correct, ref_broken);
        }
    }

    // Manually constructed block: scale=2, zero=1.5, all codes=4, q8 scale=0.25, all q8 codes=1
    // Correct: d8/scale * 32 * (4*1 - 1.5*1) = 0.25/2 * 32 * 2.5 = 10.0
    // Int-zero: 0.25/2 * 32 * (4 - 1*1) = 0.125 * 32 * 3 = 12.0
    {
        const int n = 32;
        // Q4_HQQ block as raw bytes: 20 bytes
        std::vector<uint8_t> manual_q4(20);
        ggml_fp16_t scale_h = ggml_fp32_to_fp16(2.0f);
        ggml_fp16_t zero_h  = ggml_fp32_to_fp16(1.5f);
        memcpy(&manual_q4[0], &scale_h, 2);
        memcpy(&manual_q4[2], &zero_h,  2);
        memset(&manual_q4[4], 0x44, 16); // low and high nibble both 4

        // Q8_0 block as raw bytes: 34 bytes
        std::vector<uint8_t> manual_q8(34);
        ggml_fp16_t d_h = ggml_fp32_to_fp16(0.25f);
        memcpy(&manual_q8[0], &d_h, 2);
        memset(&manual_q8[2], 1, 32);

        float dot_via_vec = 0.0f;
        const auto * qfns_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);
        qfns_cpu->vec_dot(n, &dot_via_vec, 0, manual_q4.data(), 0, manual_q8.data(), 0, 1);

        float expected = 10.0f;
        float broken   = 12.0f;
        float err = fabsf(dot_via_vec - expected);
        float err_broken = fabsf(dot_via_vec - broken);
        bool ok = (err < 1.0e-4f) && (err_broken > 1.0f);
        if (!ok) {
            printf("manual-block: vec_dot=%f expected=%f broken=%f: FAILED\n", dot_via_vec, expected, broken);
            num_failed++;
        } else {
            printf("%-40s vec_dot=%f expected=%f broken=%f: ok\n", "dot manual constant block (int-zero fail)",
                   dot_via_vec, expected, broken);
        }
    }

    // ggml_validate_row_data tests
    {
        int bad = 0;
        const int block_size = 32;
        size_t nbytes = ggml_row_size(GGML_TYPE_Q4_HQQ, block_size);
        std::vector<uint8_t> buf(nbytes, 0);

        // 1. normal block: scale=2.0, zero=1.5, codes=0x44 (all 4)
        {
            ggml_fp16_t scale = ggml_fp32_to_fp16(2.0f);
            ggml_fp16_t zero  = ggml_fp32_to_fp16(1.5f);
            memcpy(&buf[0], &scale, 2);
            memcpy(&buf[2], &zero,  2);
            memset(&buf[4], 0x44, 16);
            if (!ggml_validate_row_data(GGML_TYPE_Q4_HQQ, buf.data(), nbytes)) {
                printf("validate: normal block REJECTED: FAILED\n");
                bad++;
            } else {
                printf("%-40s: ok\n", "validate normal block");
            }
        }

        // 2. zero scale: must be rejected
        {
            ggml_fp16_t scale = ggml_fp32_to_fp16(0.0f);
            ggml_fp16_t zero  = ggml_fp32_to_fp16(1.5f);
            memcpy(&buf[0], &scale, 2);
            memcpy(&buf[2], &zero,  2);
            memset(&buf[4], 0x44, 16);
            if (ggml_validate_row_data(GGML_TYPE_Q4_HQQ, buf.data(), nbytes)) {
                printf("validate: zero scale ACCEPTED: FAILED\n");
                bad++;
            } else {
                printf("%-40s: ok\n", "validate zero scale rejected");
            }
        }

        // 3. Inf scale: must be rejected
        {
            uint16_t inf_bits = 0x7C00; // +Inf in FP16
            memcpy(&buf[0], &inf_bits, 2);
            ggml_fp16_t zero  = ggml_fp32_to_fp16(1.5f);
            memcpy(&buf[2], &zero,  2);
            memset(&buf[4], 0x44, 16);
            if (ggml_validate_row_data(GGML_TYPE_Q4_HQQ, buf.data(), nbytes)) {
                printf("validate: Inf scale ACCEPTED: FAILED\n");
                bad++;
            } else {
                printf("%-40s: ok\n", "validate Inf scale rejected");
            }
        }

        // 4. Inf zero: must be rejected
        {
            ggml_fp16_t scale = ggml_fp32_to_fp16(2.0f);
            uint16_t inf_bits = 0x7C00; // +Inf in FP16
            memcpy(&buf[0], &scale, 2);
            memcpy(&buf[2], &inf_bits, 2);
            memset(&buf[4], 0x44, 16);
            if (ggml_validate_row_data(GGML_TYPE_Q4_HQQ, buf.data(), nbytes)) {
                printf("validate: Inf zero ACCEPTED: FAILED\n");
                bad++;
            } else {
                printf("%-40s: ok\n", "validate Inf zero rejected");
            }
        }

        // 5. NaN zero: must be rejected
        {
            ggml_fp16_t scale = ggml_fp32_to_fp16(2.0f);
            uint16_t nan_bits = 0x7E00; // quiet NaN in FP16
            memcpy(&buf[0], &scale, 2);
            memcpy(&buf[2], &nan_bits, 2);
            memset(&buf[4], 0x44, 16);
            if (ggml_validate_row_data(GGML_TYPE_Q4_HQQ, buf.data(), nbytes)) {
                printf("validate: NaN zero ACCEPTED: FAILED\n");
                bad++;
            } else {
                printf("%-40s: ok\n", "validate NaN zero rejected");
            }
        }

        // 6. All-zero block sentinel: zero scale+zero+codes must still be rejected
        //    (sentinel is only for runtime, not for persisted GGUF)
        {
            memset(&buf[0], 0, nbytes);
            if (ggml_validate_row_data(GGML_TYPE_Q4_HQQ, buf.data(), nbytes)) {
                printf("validate: all-zero block ACCEPTED (should reject): FAILED\n");
                bad++;
            } else {
                printf("%-40s: ok\n", "validate all-zero block rejected (sentinel not persisted)");
            }
        }

        num_failed += (bad > 0);
    }

    //
    // Zero-buffer sentinel regression tests
    //
    // KV cache buffers are initialized by byte-wise zeroing, producing
    // scale=0, zero=0, all qs=0 blocks.  The sentinel treats this pattern
    // as all-zero output without the 0/0 NaN problem.

    // 1. All-zero block dequantization  ->  every output finite and exactly zero
    {
        const int qk = 32;
        int bad = 0;
        size_t nbytes = ggml_row_size(GGML_TYPE_Q4_HQQ, qk);
        std::vector<uint8_t> zero_buf(nbytes, 0);
        std::vector<float> dequant(qk);

        const auto * qfns = ggml_get_type_traits(GGML_TYPE_Q4_HQQ);
        qfns->to_float(zero_buf.data(), dequant.data(), qk);

        for (int i = 0; i < qk; i++) {
            if (!isfinite(dequant[i])) {
                printf("zero-block dequant: element %d is not finite (%f): FAILED\n", i, dequant[i]);
                bad++;
            }
            if (dequant[i] != 0.0f) {
                printf("zero-block dequant: element %d is %f, expected 0: FAILED\n", i, dequant[i]);
                bad++;
            }
        }
        printf("%-40s: %s\n", "zero-block dequant all finite zero", RESULT_STR[bad > 0]);
        num_failed += (bad > 0);
    }

    // 2. All-zero block dot product with non-zero Q8_0  ->  result finite and exactly zero
    {
        const int qk = 32;
        int bad = 0;

        // All-zero Q4_HQQ block
        size_t row_size_q4 = ggml_row_size(GGML_TYPE_Q4_HQQ, qk);
        std::vector<uint8_t> zero_q4(row_size_q4, 0);

        // Non-zero Q8_0 block: scale=1.0, all codes=1
        size_t row_size_q8 = ggml_row_size(GGML_TYPE_Q8_0, qk);
        std::vector<uint8_t> q8_buf(row_size_q8);
        ggml_fp16_t d8 = ggml_fp32_to_fp16(1.0f);
        memcpy(q8_buf.data(), &d8, 2);
        memset(q8_buf.data() + 2, 1, qk);

        float dot_result = 1234.0f; // sentinel
        const auto * cpu_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);
        cpu_traits->vec_dot(qk, &dot_result, 0, zero_q4.data(), 0, q8_buf.data(), 0, 1);

        if (!isfinite(dot_result)) {
            printf("zero-block vec_dot: result not finite (%f): FAILED\n", dot_result);
            bad++;
        }
        if (dot_result != 0.0f) {
            printf("zero-block vec_dot: result is %f, expected 0: FAILED\n", dot_result);
            bad++;
        }
        printf("%-40s: %s\n", "zero-block vec_dot finite zero", RESULT_STR[bad > 0]);
        num_failed += (bad > 0);
    }

    // 3. Normal valid block still uses the required formula
    {
        const int qk = 32;
        int bad = 0;
        size_t row_size_q4 = ggml_row_size(GGML_TYPE_Q4_HQQ, qk);
        std::vector<uint8_t> buf(row_size_q4);

        // scale=2.0, zero=1.5, codes=0x44 (all 4)
        ggml_fp16_t scale = ggml_fp32_to_fp16(2.0f);
        ggml_fp16_t zero  = ggml_fp32_to_fp16(1.5f);
        memcpy(&buf[0], &scale, 2);
        memcpy(&buf[2], &zero,  2);
        memset(&buf[4], 0x44, 16);

        // Dequantize
        std::vector<float> dequant(qk);
        const auto * qfns = ggml_get_type_traits(GGML_TYPE_Q4_HQQ);
        qfns->to_float(buf.data(), dequant.data(), qk);

        float expected = ((float)4 - 1.5f) / 2.0f; // = 1.25
        for (int i = 0; i < qk; i++) {
            if (fabsf(dequant[i] - expected) > 1.0e-4f) {
                printf("normal-block dequant: element %d is %f, expected %f: FAILED\n", i, dequant[i], expected);
                bad++;
                break;
            }
        }

        // Dot product with Q8_0 scale=0.25, all codes=1  ->  expected = 10.0
        size_t row_size_q8 = ggml_row_size(GGML_TYPE_Q8_0, qk);
        std::vector<uint8_t> q8_buf(row_size_q8);
        ggml_fp16_t d8 = ggml_fp32_to_fp16(0.25f);
        memcpy(q8_buf.data(), &d8, 2);
        memset(q8_buf.data() + 2, 1, qk);

        float dot_result = 0.0f;
        const auto * cpu_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);
        cpu_traits->vec_dot(qk, &dot_result, 0, buf.data(), 0, q8_buf.data(), 0, 1);

        float dot_expected = 0.25f / 2.0f * ((float)(32*4) - 1.5f * 32.0f); // = 10.0
        if (fabsf(dot_result - dot_expected) > 1.0e-4f) {
            printf("normal-block vec_dot: result %f, expected %f: FAILED\n", dot_result, dot_expected);
            bad++;
        }

        printf("%-40s: %s\n", "normal block formula preserved", RESULT_STR[bad > 0]);
        num_failed += (bad > 0);
    }

    // 4. Malformed zero-scale block (non-zero codes/metadata) still rejected
    {
        size_t nbytes = ggml_row_size(GGML_TYPE_Q4_HQQ, 32);
        std::vector<uint8_t> buf(nbytes, 0);
        ggml_fp16_t scale = ggml_fp32_to_fp16(0.0f);
        ggml_fp16_t zero  = ggml_fp32_to_fp16(1.5f);
        memcpy(&buf[0], &scale, 2);
        memcpy(&buf[2], &zero,  2);
        memset(&buf[4], 0x44, 16);
        int bad = 0;
        if (ggml_validate_row_data(GGML_TYPE_Q4_HQQ, buf.data(), nbytes)) {
            printf("validate: zero-scale with non-zero codes ACCEPTED: FAILED\n");
            bad++;
        } else {
            printf("%-40s: ok\n", "validate zero-scale non-zero codes rejected");
        }
        num_failed += (bad > 0);
    }

    if (num_failed) {
        printf("\n%d test(s) FAILED\n", num_failed);
    } else {
        printf("\nAll tests PASSED\n");
    }

    return num_failed > 0 ? 1 : 0;
}