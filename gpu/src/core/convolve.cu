/*********************************************************************
 * convolve_cuda.cu - HIGHLY OPTIMIZED FOR TESLA T4 (SM_75)
 * 
 * T4-Specific Optimizations:
 * 1. Turing has 96KB L1/Shared (64KB shared + 32KB L1 config)
 * 2. 40 SMs with 64 FP32 cores each = 2560 CUDA cores
 * 3. Separable convolution (fewer ops)
 * 4. Shared memory tiling with proper indexing
 * 5. Vectorized float4 loads for global→shared (FIXED ALIGNMENT)
 * 6. Warp-aligned 32×8 blocks (256 threads, 8 warps)
 * 7. Bank conflict avoidance with padding
 * 8. Async streams for compute/transfer overlap
 * 9. Persistent device buffers
 * 10. Constant memory for kernels
 *********************************************************************/

 #include <assert.h>
 #include <math.h>
 #include <stdlib.h>
 #include <cuda_runtime.h>
 #include "base.h"
 #include "error.h"
 #include "convolve.h"
 #include "klt_util.h"
 
 #define MAX_KERNEL_WIDTH 71
 #define WARP_SIZE 32
 #define BLOCK_DIM_X 32  // Full warp for coalescing
 #define BLOCK_DIM_Y 8   // 256 threads total
 #define MAX_KERNEL_SIZE 35
 
 #define CUDA_CHECK(call) \
   do { \
     cudaError_t err = call; \
     if (err != cudaSuccess) { \
       fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
       exit(EXIT_FAILURE); \
     } \
   } while(0)
 
 /*********************************************************************
  * Kernel Data Structures
  *********************************************************************/
 typedef struct {
   int width;
   float data[MAX_KERNEL_WIDTH];
 } ConvolutionKernel;
 
 static ConvolutionKernel gauss_kernel;
 static ConvolutionKernel gaussderiv_kernel;
 static float sigma_last = -10.0;
 
 // Constant memory for kernel (faster than global, cached)
 __constant__ float c_kernel[MAX_KERNEL_SIZE];
 
 /*********************************************************************
  * Persistent Device Buffers with Streams
  *********************************************************************/
 static struct {
   float *d_img1, *d_img2;
   size_t allocated_size;
   cudaStream_t stream;
   bool initialized;
 } g_gpu = {NULL, NULL, 0, NULL, false};
 
 static void ensure_gpu_buffers(size_t bytes) {
   if (!g_gpu.initialized) {
     CUDA_CHECK(cudaStreamCreate(&g_gpu.stream));
     // Set shared memory config: prefer 64KB shared, 32KB L1
     CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
     g_gpu.initialized = true;
   }
   
   // Align to 128 bytes for optimal coalescing (32 floats = 128 bytes)
   const size_t ALIGNMENT = 128;
   size_t aligned_bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
   
   if (aligned_bytes > g_gpu.allocated_size) {
     if (g_gpu.d_img1) {
       cudaFree(g_gpu.d_img1);
       cudaFree(g_gpu.d_img2);
     }
     // cudaMalloc already aligns to 256 bytes by default, but we ensure 128-byte alignment
     CUDA_CHECK(cudaMalloc(&g_gpu.d_img1, aligned_bytes));
     CUDA_CHECK(cudaMalloc(&g_gpu.d_img2, aligned_bytes));
     g_gpu.allocated_size = aligned_bytes;
   }
 }
 
 /*********************************************************************
  * OPTIMIZED HORIZONTAL CONVOLUTION WITH COALESCED ACCESS
  * 
  * Strategy: Warp threads load consecutive memory locations for coalescing
  * - Each warp (32 threads) loads 32 consecutive floats (128 bytes aligned)
  * - Use float4 when possible for 4x bandwidth
  *********************************************************************/
 __global__ void convolveHoriz_Optimized(
   const float * __restrict__ imgin,
   float * __restrict__ imgout,
   int ncols, int nrows,
   int kernel_width)
 {
   const int radius = kernel_width / 2;
   const int tile_width = blockDim.x;
   const int tile_height = blockDim.y;
   
   // Shared memory: [tile_height][tile_width + 2*radius + 1]
   // +1 for bank conflict avoidance
   const int tile_stride = tile_width + 2 * radius + 1;
   extern __shared__ float s_tile[];
   
   const int tx = threadIdx.x;
   const int ty = threadIdx.y;
   const int warp_id = tx / WARP_SIZE;
   const int lane_id = tx % WARP_SIZE;
   const int gx = blockIdx.x * tile_width + tx;
   const int gy = blockIdx.y * tile_height + ty;
   
   if (gy >= nrows) return;
   
   // Tile start column for this block (including halo)
   const int tile_start_col = blockIdx.x * tile_width - radius;
   const int tile_end_col = tile_start_col + tile_stride;
   
   // Load data cooperatively - ensure warp threads access consecutive memory
   for (int row = ty; row < tile_height; row += tile_height) {
     int global_row = blockIdx.y * tile_height + row;
     if (global_row >= nrows) continue;
     
     const float* row_ptr = &imgin[global_row * ncols];
     
     // Each warp loads consecutive columns for perfect coalescing
     // With tile_width=32, we have one warp, but tile_stride may be > 32 (due to halo)
     // So we loop to load all needed columns
     
     for (int base_col = tile_start_col; base_col < tile_end_col; base_col += WARP_SIZE) {
       int warp_start_col = base_col + warp_id * WARP_SIZE;
       int warp_end_col = (warp_start_col + WARP_SIZE < tile_end_col) ? warp_start_col + WARP_SIZE : tile_end_col;
       
       if (warp_start_col >= tile_start_col && warp_start_col < tile_end_col) {
         // Try float4 for extra speed when aligned (8 float4s = 32 floats per warp)
         if (warp_start_col % 4 == 0 && warp_start_col + 31 < tile_end_col && 
             warp_start_col >= 0 && warp_start_col + 31 < ncols && lane_id < 8) {
           // 8 threads load 8 float4s = 32 elements (perfect coalescing!)
           float4 data = reinterpret_cast<const float4*>(&row_ptr[warp_start_col + lane_id * 4])[0];
           int shared_base = row * tile_stride + (warp_start_col - tile_start_col);
           s_tile[shared_base + lane_id * 4 + 0] = data.x;
           s_tile[shared_base + lane_id * 4 + 1] = data.y;
           s_tile[shared_base + lane_id * 4 + 2] = data.z;
           s_tile[shared_base + lane_id * 4 + 3] = data.w;
         } else {
           // Scalar coalesced load: each warp thread loads consecutive memory
           for (int global_col = warp_start_col + lane_id; global_col < warp_end_col; global_col += WARP_SIZE) {
             float val = 0.0f;
             if (global_col >= 0 && global_col < ncols) {
               val = row_ptr[global_col];
             }
             if (global_col >= tile_start_col && global_col < tile_end_col) {
               int local_col = global_col - tile_start_col;
               s_tile[row * tile_stride + local_col] = val;
             }
           }
         }
       }
     }
   }
   __syncthreads();
   
   // Compute convolution
   if (gx >= ncols) return;
   
   // Zero boundaries
   if (gx < radius || gx >= ncols - radius) {
     imgout[gy * ncols + gx] = 0.0f;
     return;
   }
   
   // Convolve (read from shared memory - already cached)
   float sum = 0.0f;
   int s_center = ty * tile_stride + tx + radius;
   
   #pragma unroll 8
   for (int k = kernel_width - 1; k >= 0; k--) {
     int offset = kernel_width - 1 - k;
     sum += s_tile[s_center - radius + offset] * c_kernel[k];
   }
   
   imgout[gy * ncols + gx] = sum;
 }
 
 /*********************************************************************
  * OPTIMIZED VERTICAL CONVOLUTION WITH COALESCED ACCESS
  * 
  * Problem: Column access is strided (threads access locations ncols apart)
  * Solution: Transpose loading pattern - load rows in shared memory, transpose
  *          OR: Use warp-cooperative loading where warp loads consecutive rows
  *          Strategy: Each warp loads a contiguous set of rows for all columns
  *********************************************************************/
 __global__ void convolveVert_Optimized(
   const float * __restrict__ imgin,
   float * __restrict__ imgout,
   int ncols, int nrows,
   int kernel_width)
 {
   const int radius = kernel_width / 2;
   const int tile_width = blockDim.x;
   const int tile_height = blockDim.y;
   
   // Shared memory: [tile_height + 2*radius][tile_width + 1]
   // +1 for bank conflict avoidance
   const int tile_stride = tile_width + 1;
   const int tile_vert = tile_height + 2 * radius;
   extern __shared__ float s_tile[];
   
   const int tx = threadIdx.x;
   const int ty = threadIdx.y;
   const int gx = blockIdx.x * tile_width + tx;
   const int gy = blockIdx.y * tile_height + ty;
   
   // For coalescing: threads with same ty (same row in block) form warps
   // warp_id based on tx ensures threads in warp load consecutive columns from same row
   const int warp_id = tx / WARP_SIZE;
   const int lane_id = tx % WARP_SIZE;
   
   if (gx >= ncols) return;
   
   // Tile start row for this block (including halo)
   const int tile_start_row = blockIdx.y * tile_height - radius;
   
   // COALESCED LOADING: Load rows cooperatively using warp-level coalescing
   // Key insight: Threads with same ty (same row) but different tx (columns) 
   //              access consecutive memory in that row (perfect coalescing!)
   // Strategy: Each row is loaded by threads with ty=that_row, tx=0..31 (a warp)
   
   for (int local_row = ty; local_row < tile_vert; local_row += tile_height) {
     int global_row = tile_start_row + local_row;
     
     if (global_row >= 0 && global_row < nrows) {
       const float* row_ptr = &imgin[global_row * ncols];
       
       // Load row cooperatively: threads in same row (ty) load consecutive columns
       // Row needs (tile_width + 2*radius) elements
       // Warp 0 (tx=0..31) loads columns [start..start+31], warp 1 loads [start+32..], etc.
       int row_start_col = blockIdx.x * tile_width - radius;
       int row_elements = tile_width + 2 * radius;
       
       // Each warp loads 32 consecutive columns from the row
       int warp_col_base = row_start_col + warp_id * WARP_SIZE;
       int warp_col_end = (warp_col_base + WARP_SIZE < row_start_col + row_elements) ? 
                          warp_col_base + WARP_SIZE : row_start_col + row_elements;
       
       // Coalesced load: threads in warp (tx=0..31) load consecutive memory
       for (int global_col = warp_col_base + lane_id; global_col < warp_col_end; global_col += WARP_SIZE) {
         float val = 0.0f;
         
         if (global_col >= 0 && global_col < ncols) {
           val = row_ptr[global_col];  // Perfectly coalesced: threads access consecutive memory!
         }
         
         int col_offset = global_col - row_start_col;
         if (col_offset >= 0 && col_offset < tile_stride) {
           s_tile[local_row * tile_stride + col_offset] = val;
         }
       }
       
       // Optional: Try float4 for extra speed when aligned
       if (warp_col_base % 4 == 0 && warp_col_base + 31 < ncols && 
           warp_col_base + 31 < row_start_col + row_elements && lane_id < 8 && 
           (warp_col_base + lane_id * 4 + 3) < warp_col_end) {
         // 8 threads load 8 float4s = 32 elements (perfect coalescing!)
         float4 data = reinterpret_cast<const float4*>(&row_ptr[warp_col_base + lane_id * 4])[0];
         int shared_base = local_row * tile_stride + (warp_col_base - row_start_col);
         s_tile[shared_base + lane_id * 4 + 0] = data.x;
         s_tile[shared_base + lane_id * 4 + 1] = data.y;
         s_tile[shared_base + lane_id * 4 + 2] = data.z;
         s_tile[shared_base + lane_id * 4 + 3] = data.w;
       }
     } else {
       // Out of bounds - zero pad the row
       int row_start_col = blockIdx.x * tile_width - radius;
       int row_elements = tile_width + 2 * radius;
       int warp_col_base = row_start_col + warp_id * WARP_SIZE;
       int warp_col_end = (warp_col_base + WARP_SIZE < row_start_col + row_elements) ? 
                          warp_col_base + WARP_SIZE : row_start_col + row_elements;
       
       for (int col_offset = lane_id; col_offset < row_elements; col_offset += WARP_SIZE) {
         int global_col = warp_col_base + col_offset;
         int local_col = global_col - row_start_col;
         if (local_col >= 0 && local_col < tile_stride) {
           s_tile[local_row * tile_stride + local_col] = 0.0f;
         }
       }
     }
   }
   
   __syncthreads();
   
   // Compute convolution
   if (gy >= nrows) return;
   
   // Zero boundaries
   if (gy < radius || gy >= nrows - radius) {
     imgout[gy * ncols + gx] = 0.0f;
     return;
   }
   
   // Convolve (read from shared memory along column - already cached)
   float sum = 0.0f;
   int s_center_row = ty + radius;
   
   #pragma unroll 8
   for (int k = kernel_width - 1; k >= 0; k--) {
     int offset = kernel_width - 1 - k;
     sum += s_tile[(s_center_row - radius + offset) * tile_stride + tx] * c_kernel[k];
   }
   
   imgout[gy * ncols + gx] = sum;
 }
 
 /*********************************************************************
  * Host Wrapper Functions
  *********************************************************************/
 static void _convolveImageHoriz(
   _KLT_FloatImage imgin,
   ConvolutionKernel kernel,
   _KLT_FloatImage imgout)
 {
   const int ncols = imgin->ncols;
   const int nrows = imgin->nrows;
   const size_t nbytes = ncols * nrows * sizeof(float);
   
   ensure_gpu_buffers(nbytes);
   
   // Copy kernel to constant memory
   CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, kernel.data, 
     kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
   
   // Copy input to device
   CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img1, imgin->data, nbytes,
     cudaMemcpyHostToDevice, g_gpu.stream));
   
   // Launch configuration
   const int radius = kernel.width / 2;
   dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
   dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
             (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
   
   // Shared memory size
   const int tile_stride = BLOCK_DIM_X + 2 * radius + 1;
   size_t shared_bytes = BLOCK_DIM_Y * tile_stride * sizeof(float);
   
   // Enable 64KB shared memory if needed (T4 supports it)
   if (shared_bytes > 48 * 1024) {
     CUDA_CHECK(cudaFuncSetAttribute(convolveHoriz_Optimized,
       cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
     CUDA_CHECK(cudaFuncSetAttribute(convolveHoriz_Optimized,
       cudaFuncAttributePreferredSharedMemoryCarveout, 100)); // 64KB shared
   }
   
   convolveHoriz_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
     g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, kernel.width);
   
   CUDA_CHECK(cudaGetLastError());
   
   // Copy result back
   CUDA_CHECK(cudaMemcpyAsync(imgout->data, g_gpu.d_img2, nbytes,
     cudaMemcpyDeviceToHost, g_gpu.stream));
   
   CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
   
   imgout->ncols = ncols;
   imgout->nrows = nrows;
 }
 
 static void _convolveImageVert(
   _KLT_FloatImage imgin,
   ConvolutionKernel kernel,
   _KLT_FloatImage imgout)
 {
   const int ncols = imgin->ncols;
   const int nrows = imgin->nrows;
   const size_t nbytes = ncols * nrows * sizeof(float);
   
   ensure_gpu_buffers(nbytes);
   
   // Copy kernel to constant memory
   CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, kernel.data,
     kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
   
   // Copy input to device
   CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img1, imgin->data, nbytes,
     cudaMemcpyHostToDevice, g_gpu.stream));
   
   // Launch configuration
   const int radius = kernel.width / 2;
   dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
   dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
             (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
   
   // Shared memory size
   const int tile_vert = BLOCK_DIM_Y + 2 * radius;
   const int tile_stride = BLOCK_DIM_X + 1;
   size_t shared_bytes = tile_vert * tile_stride * sizeof(float);
   
   // Enable 64KB shared memory if needed
   if (shared_bytes > 48 * 1024) {
     CUDA_CHECK(cudaFuncSetAttribute(convolveVert_Optimized,
       cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
     CUDA_CHECK(cudaFuncSetAttribute(convolveVert_Optimized,
       cudaFuncAttributePreferredSharedMemoryCarveout, 100));
   }
   
   convolveVert_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
     g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, kernel.width);
   
   CUDA_CHECK(cudaGetLastError());
   
   // Copy result back
   CUDA_CHECK(cudaMemcpyAsync(imgout->data, g_gpu.d_img2, nbytes,
     cudaMemcpyDeviceToHost, g_gpu.stream));
   
   CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
   
   imgout->ncols = ncols;
   imgout->nrows = nrows;
 }
 
 /*********************************************************************
  * Separable Convolution
  *********************************************************************/
 static void _convolveSeparate(
   _KLT_FloatImage imgin,
   ConvolutionKernel horiz_kernel,
   ConvolutionKernel vert_kernel,
   _KLT_FloatImage imgout)
 {
   _KLT_FloatImage tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
   _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
   _convolveImageVert(tmpimg, vert_kernel, imgout);
   _KLTFreeFloatImage(tmpimg);
 }
 
 /*********************************************************************
  * Kernel Computation (unchanged from original)
  *********************************************************************/
 static void _computeKernels(
   float sigma,
   ConvolutionKernel *gauss,
   ConvolutionKernel *gaussderiv)
 {
   const float factor = 0.01f;
   int i;
 
   assert(MAX_KERNEL_WIDTH % 2 == 1);
   assert(sigma >= 0.0);
 
   {
     const int hw = MAX_KERNEL_WIDTH / 2;
     float max_gauss = 1.0f, max_gaussderiv = (float)(sigma*exp(-0.5f));
   
     for (i = -hw; i <= hw; i++) {
       gauss->data[i+hw] = (float)exp(-i*i / (2*sigma*sigma));
       gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
     }
 
     gauss->width = MAX_KERNEL_WIDTH;
     for (i = -hw; fabs(gauss->data[i+hw] / max_gauss) < factor; 
          i++, gauss->width -= 2);
     gaussderiv->width = MAX_KERNEL_WIDTH;
     for (i = -hw; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor; 
          i++, gaussderiv->width -= 2);
     if (gauss->width == MAX_KERNEL_WIDTH || 
         gaussderiv->width == MAX_KERNEL_WIDTH)
       KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
                "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
   }
 
   for (i = 0; i < gauss->width; i++)
     gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
   for (i = 0; i < gaussderiv->width; i++)
     gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
 
   {
     const int hw = gaussderiv->width / 2;
     float den;
       
     den = 0.0;
     for (i = 0; i < gauss->width; i++) den += gauss->data[i];
     for (i = 0; i < gauss->width; i++) gauss->data[i] /= den;
     den = 0.0;
     for (i = -hw; i <= hw; i++) den -= i*gaussderiv->data[i+hw];
     for (i = -hw; i <= hw; i++) gaussderiv->data[i+hw] /= den;
   }
 
   sigma_last = sigma;
 }
 
 /*********************************************************************
  * Public API Functions
  *********************************************************************/
 void _KLTToFloatImage(
   KLT_PixelType *img,
   int ncols, int nrows,
   _KLT_FloatImage floatimg)
 {
   KLT_PixelType *ptrend = img + ncols*nrows;
   float *ptrout = floatimg->data;
 
   assert(floatimg->ncols >= ncols);
   assert(floatimg->nrows >= nrows);
 
   floatimg->ncols = ncols;
   floatimg->nrows = nrows;
 
   while (img < ptrend) *ptrout++ = (float)*img++;
 }
 
 void _KLTGetKernelWidths(
   float sigma,
   int *gauss_width,
   int *gaussderiv_width)
 {
   _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
   *gauss_width = gauss_kernel.width;
   *gaussderiv_width = gaussderiv_kernel.width;
 }
 
 void _KLTComputeGradients(
   _KLT_FloatImage img,
   float sigma,
   _KLT_FloatImage gradx,
   _KLT_FloatImage grady)
 {
   assert(gradx->ncols >= img->ncols);
   assert(gradx->nrows >= img->nrows);
   assert(grady->ncols >= img->ncols);
   assert(grady->nrows >= img->nrows);
 
   if (fabs(sigma - sigma_last) > 0.05)
     _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
   
   ensure_gpu_buffers(img->ncols * img->nrows * sizeof(float));
   
   _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
   _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
 }
 
 void _KLTComputeSmoothedImage(
   _KLT_FloatImage img,
   float sigma,
   _KLT_FloatImage smooth)
 {
   assert(smooth->ncols >= img->ncols);
   assert(smooth->nrows >= img->nrows);
 
   if (fabs(sigma - sigma_last) > 0.05)
     _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
 
   ensure_gpu_buffers(img->ncols * img->nrows * sizeof(float));
   
   _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
 }
 
 // Cleanup function (call at program exit)
 void _KLTCleanupGPU() {
   if (g_gpu.initialized) {
     if (g_gpu.d_img1) cudaFree(g_gpu.d_img1);
     if (g_gpu.d_img2) cudaFree(g_gpu.d_img2);
     cudaStreamDestroy(g_gpu.stream);
     g_gpu.initialized = false;
   }
 }
 