#include <cstdint>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define GF137_USE_ARM_NEON 1
#else
#define GF137_USE_ARM_NEON 0
#endif

namespace {

constexpr uint32_t kModulus = 137U;
constexpr uint64_t kBarrettMu = (uint64_t{1} << 32) / kModulus;

inline uint8_t mod137_barrett(uint32_t value) {
    const uint32_t quotient = static_cast<uint32_t>(
        (static_cast<uint64_t>(value) * kBarrettMu) >> 32
    );
    uint32_t residue = value - quotient * kModulus;
    while (residue >= kModulus) {
        residue -= kModulus;
    }
    return static_cast<uint8_t>(residue);
}

#if GF137_USE_ARM_NEON
inline uint16x8_t mod137_u16x8(uint16x8_t values) {
    const uint16x8_t modulus = vdupq_n_u16(137U);
    const uint32x4_t product_low = vmull_n_u16(vget_low_u16(values), 478U);
    const uint32x4_t product_high = vmull_n_u16(vget_high_u16(values), 478U);
    const uint16x8_t quotient = vcombine_u16(
        vshrn_n_u32(product_low, 16),
        vshrn_n_u32(product_high, 16)
    );
    uint16x8_t residue = vsubq_u16(values, vmulq_u16(quotient, modulus));
    const uint16x8_t ge_modulus = vcgeq_u16(residue, modulus);
    residue = vsubq_u16(residue, vandq_u16(ge_modulus, modulus));
    return residue;
}

inline uint32_t active_weight_sum_u16x8(
    uint16x8_t acc,
    const uint8_t* weights,
    uint8_t threshold
) {
    const uint16x8_t residue = mod137_u16x8(acc);
    const uint16x8_t active = vandq_u16(
        vcgeq_u16(residue, vdupq_n_u16(static_cast<uint16_t>(threshold))),
        vdupq_n_u16(1U)
    );
    const uint16x8_t w = vmovl_u8(vld1_u8(weights));
    return static_cast<uint32_t>(vaddvq_u16(vmulq_u16(active, w)));
}

inline uint8_t predict_one_neon32(
    const uint8_t* x,
    const uint8_t* w1,
    const uint8_t* b1,
    const uint8_t* w2,
    uint8_t b2,
    int k_width,
    uint8_t hidden_threshold,
    uint8_t output_threshold
) {
    uint16x8_t acc0 = vmovl_u8(vld1_u8(b1));
    uint16x8_t acc1 = vmovl_u8(vld1_u8(b1 + 8));
    uint16x8_t acc2 = vmovl_u8(vld1_u8(b1 + 16));
    uint16x8_t acc3 = vmovl_u8(vld1_u8(b1 + 24));

    for (int k = 0; k < k_width; ++k) {
        const uint8_t x_value = x[k];
        if (x_value == 0U) {
            continue;
        }
        const uint8_t* w_row = w1 + static_cast<int64_t>(k) * 32;
        const uint16x8_t w0 = vmovl_u8(vld1_u8(w_row));
        const uint16x8_t w1v = vmovl_u8(vld1_u8(w_row + 8));
        const uint16x8_t w2v = vmovl_u8(vld1_u8(w_row + 16));
        const uint16x8_t w3v = vmovl_u8(vld1_u8(w_row + 24));
        if (x_value == 1U) {
            acc0 = vaddq_u16(acc0, w0);
            acc1 = vaddq_u16(acc1, w1v);
            acc2 = vaddq_u16(acc2, w2v);
            acc3 = vaddq_u16(acc3, w3v);
        } else {
            acc0 = vaddq_u16(acc0, vshlq_n_u16(w0, 1));
            acc1 = vaddq_u16(acc1, vshlq_n_u16(w1v, 1));
            acc2 = vaddq_u16(acc2, vshlq_n_u16(w2v, 1));
            acc3 = vaddq_u16(acc3, vshlq_n_u16(w3v, 1));
        }
    }

    uint32_t out_acc = static_cast<uint32_t>(b2);
    out_acc += active_weight_sum_u16x8(acc0, w2, hidden_threshold);
    out_acc += active_weight_sum_u16x8(acc1, w2 + 8, hidden_threshold);
    out_acc += active_weight_sum_u16x8(acc2, w2 + 16, hidden_threshold);
    out_acc += active_weight_sum_u16x8(acc3, w2 + 24, hidden_threshold);
    return static_cast<uint8_t>(mod137_barrett(out_acc) >= output_threshold ? 1U : 0U);
}
#endif

inline uint8_t predict_one_portable(
    const uint8_t* x,
    const uint8_t* w1,
    const uint8_t* b1,
    const uint8_t* w2,
    uint8_t b2,
    int k_width,
    int h_width,
    uint8_t hidden_threshold,
    uint8_t output_threshold
) {
    uint32_t hidden_acc[128];
    for (int h = 0; h < h_width; ++h) {
        hidden_acc[h] = static_cast<uint32_t>(b1[h]);
    }
    for (int k = 0; k < k_width; ++k) {
        const uint32_t x_value = static_cast<uint32_t>(x[k]);
        if (x_value == 0U) {
            continue;
        }
        const uint8_t* w_row = w1 + static_cast<int64_t>(k) * h_width;
        if (x_value == 1U) {
            for (int h = 0; h < h_width; ++h) {
                hidden_acc[h] += static_cast<uint32_t>(w_row[h]);
            }
        } else {
            for (int h = 0; h < h_width; ++h) {
                hidden_acc[h] += 2U * static_cast<uint32_t>(w_row[h]);
            }
        }
    }

    uint32_t out_acc = static_cast<uint32_t>(b2);
    for (int h = 0; h < h_width; ++h) {
        const uint8_t hidden_residue = mod137_barrett(hidden_acc[h]);
        if (hidden_residue >= hidden_threshold) {
            out_acc += static_cast<uint32_t>(w2[h]);
        }
    }
    return static_cast<uint8_t>(mod137_barrett(out_acc) >= output_threshold ? 1U : 0U);
}

void predict_batch(
    const uint8_t* x,
    const uint8_t* w1,
    const uint8_t* b1,
    const uint8_t* w2,
    const uint8_t* b2,
    uint8_t* y,
    int rows,
    int k_width,
    int h_width,
    uint8_t hidden_threshold,
    uint8_t output_threshold
) {
    for (int row = 0; row < rows; ++row) {
#if GF137_USE_ARM_NEON
        if (h_width == 32) {
            y[row] = predict_one_neon32(
                x + static_cast<int64_t>(row) * k_width,
                w1,
                b1,
                w2,
                b2[0],
                k_width,
                hidden_threshold,
                output_threshold
            );
            continue;
        }
#endif
        y[row] = predict_one_portable(
            x + static_cast<int64_t>(row) * k_width,
            w1,
            b1,
            w2,
            b2[0],
            k_width,
            h_width,
            hidden_threshold,
            output_threshold
        );
    }
}

}  // namespace

extern "C" {

void gf137_predict(
    const uint8_t* x,
    const uint8_t* w1,
    const uint8_t* b1,
    const uint8_t* w2,
    const uint8_t* b2,
    uint8_t* y,
    int rows,
    int k_width,
    int h_width,
    uint8_t hidden_threshold,
    uint8_t output_threshold
) {
    predict_batch(
        x,
        w1,
        b1,
        w2,
        b2,
        y,
        rows,
        k_width,
        h_width,
        hidden_threshold,
        output_threshold
    );
}

void gf137_predict_repeated(
    const uint8_t* x,
    const uint8_t* w1,
    const uint8_t* b1,
    const uint8_t* w2,
    const uint8_t* b2,
    uint8_t* y,
    int rows,
    int k_width,
    int h_width,
    uint8_t hidden_threshold,
    uint8_t output_threshold,
    int repeats
) {
    for (int repeat = 0; repeat < repeats; ++repeat) {
        predict_batch(
            x,
            w1,
            b1,
            w2,
            b2,
            y,
            rows,
            k_width,
            h_width,
            hidden_threshold,
            output_threshold
        );
    }
}

}
