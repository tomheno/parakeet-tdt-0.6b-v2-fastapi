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
#include <assert.h>

/*
 * Zero-pad and copy variable-length float arrays into a (possibly strided) 2D buffer.
 *
 * dst:         destination buffer
 * max_len:     number of columns to zero-pad per row
 * row_stride:  actual distance (in floats) between consecutive rows in dst.
 *              This may differ from max_len when dst is a strided view of a larger buffer.
 * src_ptrs:    array of pointers to source float arrays
 * src_lengths: number of float elements in each source array
 * batch_size:  number of arrays to copy
 */
void batch_pad_copy(
    float *dst,
    int max_len,
    int row_stride,
    const float * const *src_ptrs,
    const int *src_lengths,
    int batch_size
) {
    assert(row_stride >= max_len && "row_stride must be >= max_len to prevent overlap");
    for (int i = 0; i < batch_size; i++) {
        assert(src_lengths[i] <= max_len && "src length exceeds max_len");
        float *row = dst + (size_t)i * row_stride;
        memset(row, 0, (size_t)max_len * sizeof(float));
        memcpy(row, src_ptrs[i], (size_t)src_lengths[i] * sizeof(float));
    }
}
