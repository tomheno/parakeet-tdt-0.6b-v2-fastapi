/*
 * GIL-free batch padding for audio arrays.
 *
 * ctypes releases the GIL when calling C functions, so these run truly
 * parallel with CUDA kernels on another thread.
 *
 * Compile:
 *   gcc -O3 -march=native -shared -fPIC -o _fast_copy.so _fast_copy.c
 */
#include <string.h>
#include <stddef.h>

/*
 * batch_pad_copy: scatter from pointer array (legacy).
 */
void batch_pad_copy(
    float *dst,
    int max_len,
    const float * const *src_ptrs,
    const int *src_lengths,
    int batch_size
) {
    memset(dst, 0, (size_t)batch_size * max_len * sizeof(float));
    for (int i = 0; i < batch_size; i++) {
        memcpy(
            dst + (size_t)i * max_len,
            src_ptrs[i],
            (size_t)src_lengths[i] * sizeof(float)
        );
    }
}

/*
 * batch_pad_scatter: zero-pad from a concatenated source buffer + offsets.
 *
 * No Python loop needed — caller uses np.concatenate (C) + np.cumsum (C)
 * to build src_concat and offsets, then this single C call does the scatter.
 *
 * dst:         destination [batch_size x max_len] row-major floats
 * max_len:     columns in dst
 * src_concat:  all source arrays concatenated contiguously
 * offsets:     byte offset (in floats) of each array within src_concat
 * lengths:     number of float elements in each source array
 * batch_size:  number of arrays
 */
void batch_pad_scatter(
    float *dst,
    int max_len,
    const float *src_concat,
    const long long *offsets,
    const int *lengths,
    int batch_size
) {
    memset(dst, 0, (size_t)batch_size * max_len * sizeof(float));
    for (int i = 0; i < batch_size; i++) {
        memcpy(
            dst + (size_t)i * max_len,
            src_concat + offsets[i],
            (size_t)lengths[i] * sizeof(float)
        );
    }
}
