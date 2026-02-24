/*
 * GIL-free batch padding for audio arrays.
 *
 * Replaces a Python loop of numpy slice assignments with a single C call.
 * ctypes releases the GIL when calling this, so it runs truly parallel
 * with CUDA kernels on another thread.
 *
 * Compile:
 *   gcc -O3 -march=native -shared -fPIC -o _fast_copy.so _fast_copy.c
 */
#include <string.h>
#include <stddef.h>

/*
 * Zero-pad and copy variable-length float arrays into a contiguous 2D buffer.
 *
 * dst:         destination buffer, row-major [batch_size x max_len] floats
 * max_len:     number of columns (max audio samples)
 * src_ptrs:    array of pointers to source float arrays
 * src_lengths: number of float elements in each source array
 * batch_size:  number of arrays to copy
 */
void batch_pad_copy(
    float *dst,
    int max_len,
    const float * const *src_ptrs,
    const int *src_lengths,
    int batch_size
) {
    /* Zero entire destination region */
    memset(dst, 0, (size_t)batch_size * max_len * sizeof(float));

    /* Copy each source array into its row */
    for (int i = 0; i < batch_size; i++) {
        memcpy(
            dst + (size_t)i * max_len,
            src_ptrs[i],
            (size_t)src_lengths[i] * sizeof(float)
        );
    }
}
