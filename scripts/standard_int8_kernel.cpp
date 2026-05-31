#include <cstdint>

namespace {

inline uint8_t predict_one_int8(
    const int8_t* x,
    const int8_t* w1,
    const int32_t* b1,
    const int8_t* w2,
    int32_t b2,
    int k_width,
    int h_width,
    int32_t hidden_threshold,
    int32_t output_threshold
) {
    int32_t hidden_acc[128];
    for (int h = 0; h < h_width; ++h) {
        hidden_acc[h] = b1[h];
    }

    for (int k = 0; k < k_width; ++k) {
        const int32_t x_value = static_cast<int32_t>(x[k]);
        if (x_value == 0) {
            continue;
        }
        const int8_t* w_row = w1 + static_cast<int64_t>(k) * h_width;
        for (int h = 0; h < h_width; ++h) {
            hidden_acc[h] += x_value * static_cast<int32_t>(w_row[h]);
        }
    }

    int32_t out_acc = b2;
    for (int h = 0; h < h_width; ++h) {
        if (hidden_acc[h] >= hidden_threshold) {
            out_acc += static_cast<int32_t>(w2[h]);
        }
    }
    return static_cast<uint8_t>(out_acc >= output_threshold ? 1U : 0U);
}

void predict_batch_int8(
    const int8_t* x,
    const int8_t* w1,
    const int32_t* b1,
    const int8_t* w2,
    const int32_t* b2,
    uint8_t* y,
    int rows,
    int k_width,
    int h_width,
    int32_t hidden_threshold,
    int32_t output_threshold
) {
    for (int row = 0; row < rows; ++row) {
        y[row] = predict_one_int8(
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

void standard_int8_predict_repeated(
    const int8_t* x,
    const int8_t* w1,
    const int32_t* b1,
    const int8_t* w2,
    const int32_t* b2,
    uint8_t* y,
    int rows,
    int k_width,
    int h_width,
    int32_t hidden_threshold,
    int32_t output_threshold,
    int repeats
) {
    for (int repeat = 0; repeat < repeats; ++repeat) {
        predict_batch_int8(
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
