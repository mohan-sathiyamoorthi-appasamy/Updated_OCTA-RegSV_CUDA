#include "Corr_Axial.h"
__global__ void floatDataTypeConversion(unsigned short *ushortData, float *Result, int hostfileLen)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < hostfileLen)
	{
		if (ushortData[tid] == 0)
		{
			Result[tid] = 0.0;
			
		}
		else
		{
			Result[tid] = 20 * log10f(abs((float)ushortData[tid]));
			
		}
		
		//printf("%d  %2f\n", tid, Result[tid]);
	}
	//if (tid > 487400 && tid < 487500)
	//{
		//printf("%2f\n", Result[222339]);
	//}
}

__global__ void floatTypeConversionProcess(unsigned short *ushortData, float *Result, int hostfileLen)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < hostfileLen)
	{
		if (ushortData[tid] == 0)
		{
			Result[tid] = 0.0;

		}
		else
		{
			Result[tid] = (float)ushortData[tid];

		}

		//printf("%d  %2f\n", tid, Result[tid]);
	}
	//if (tid > 487400 && tid < 487500)
	//{
	//printf("%2f\n", Result[222339]);
	//}
}


__global__ void absKernel(float *d_float_input, float *d_absInputData, int hostfileLen)
{

	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	//printf("%d\n",tid);
	if (tid < hostfileLen)
	{
		d_absInputData[tid] = abs(d_float_input[tid]);
		//printf("%2f\n", d_absInputData[tid]);
	}
}


__global__ void logKernel(float *d_absInputData, float *d_LogInputData, int hostfileLen)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < hostfileLen)
	{
		if (d_absInputData[tid] == 0)
		{
			d_LogInputData[tid] = 0.0;
		}
		else
		{
			d_LogInputData[tid] = 20 * log10f(abs(d_absInputData[tid]));
		}
		
	}
	printf("%2f\n", d_absInputData[222339]);
}


__global__ void ImageSizeModification(float *d_float_input_sizeModification, float *d_result,int imageSize)
{
	//printf("%2d\n", imageSize);
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	//printf("%2d\n", imageSize);
	if (tid < imageSize)
	{
		d_float_input_sizeModification[tid] = d_result[tid];
		//printf("%2f\n", d_float_input_sizeModification[tid]);
	}
	else
	{
		d_float_input_sizeModification[tid] = 0.0;
		//printf("%2d  %2f\n",tid, d_float_input_sizeModification[tid]);
	}

}



__global__ void block_sum(float *input, float *per_block_results, const size_t n)
{
	//printf("Kernel is Start");

	extern __shared__ float sdata[];
	long int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	// load input into __shared__ memory
	float x = 0;
	if (i < n)
	{
		x = input[i];
		//if (x<0)
			//printf("%d,%2f\n", i, input[i]);
	}
	//printf("Inf Index %2f\n",input[222339]);
	sdata[threadIdx.x] = x;
	//printf("%d, %f\n", threadIdx.x, sdata[threadIdx.x]);
	__syncthreads();


	// contiguous range pattern
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (threadIdx.x < offset)
		{
			// add a partial sum upstream to our own
			sdata[threadIdx.x] += sdata[threadIdx.x + offset];

		}
		
		//printf("%2f\n", sdata[threadIdx.x]);
		// wait until all threads in the block have
		// updated their partial sums
		__syncthreads();
		//printf("%2f\n", sdata[threadIdx.x]);
		
	}
	__syncthreads();
	//if (sdata[threadIdx.x]<0)
		//printf("%d,%2f\n", threadIdx.x ,sdata[threadIdx.x]);
	// thread 0 writes the final result
	if (threadIdx.x == 0)
	{
		per_block_results[blockIdx.x] = sdata[0];
		
		//printf("%d , %2f\n", blockIdx.x,per_block_results[blockIdx.x]);
		//printf("%2f\n", sdata[0]);
	}
}

__global__ void Subtraction(float *d_result,float *meanSubFirstImage,float device_result, int host_fileLen1,int numElements)
{
	//printf("%d\n", host_fileLen1);
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < host_fileLen1)
	{
		meanSubFirstImage[tid] = d_result[tid] - device_result;
		//printf("%2f\n",host_fileLen1);
	}
	else
	{
		meanSubFirstImage[tid] = d_result[tid] - 0.0;
	}
}


__global__ void Square(float *Square_Result_FirstImage, float *meanSubFirstImage, int host_fileLen1)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < host_fileLen1)
	{
		Square_Result_FirstImage[tid] = meanSubFirstImage[tid] * meanSubFirstImage[tid];
	}

}
__global__ void Mul_Operation(float *mulMeanSubImage, float * meanSubImage1, float* meanSubImage2, int hostfileLen2)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid < hostfileLen2)
	{
		mulMeanSubImage[tid] = meanSubImage1[tid] * meanSubImage2[tid];
	}
}

__global__ void cropMLS(float *complexArray, float *floatArray, int width, int height, int fftWidth, int frameIdx, int offset, int range)
{
	//printf("Kernel Start");
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	//Calculating the indices beforehand is much faster than calculating these indices during the function,
	//Therefore it would make sense to do all of it here first before maping the values
	int mapFloatIdx = frameIdx*range*height + idx;
	int mapCmpIdx = int(idx / range) * 975 + idx%range + offset;
	if (mapCmpIdx < (975 * 500))
	{
		floatArray[mapFloatIdx] = complexArray[mapCmpIdx];
		//printf("%d\n", mapFloatIdx);
	}
		//printf("%d, %2f\n", mapCmpIdx, complexArray[mapCmpIdx]);
}

__global__ void Modify_ImageSize_MotionCorrection(float *d_float_input, float *volume_mcorr, int new_image_Size, double top)
{
	 
	int frameIdx = 0;
	int range = 975;
	int height = 975;
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int mapCmpIdx = int(idx / 975) * 1085 + idx % 975 + round(top);

	//if (idx < (975 * 500))
	//{
	volume_mcorr[mapCmpIdx] = d_float_input[idx];
		
	//}
	//else
//	{
		//volume_mcorr[mapCmpIdx] = 0.0;
//	}
	//printf("%f\n", volume_mcorr[mapCmpIdx]);
	
}      
__global__ void find_maximum_kernel(float *array, float *max, int* maxIndex, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ float cache[256];
	__shared__ int indexCache[256];

	float temp = -1.0;
	int tempIndex = 0;
	while (index + offset < n){
		if (temp < array[index + offset])
		{
			temp = array[index + offset];
			tempIndex = index + offset;
		}  offset += stride;
	} cache[threadIdx.x] = temp;
	indexCache[threadIdx.x] = tempIndex;

	__syncthreads();

	// reduction
	unsigned int i = blockDim.x / 2;
	while (i != 0){
		if (threadIdx.x < i){
			if (cache[threadIdx.x] < cache[threadIdx.x + i]){
				cache[threadIdx.x] = cache[threadIdx.x + i];
				indexCache[threadIdx.x] = indexCache[threadIdx.x + i];
			}
		}  __syncthreads();
		i /= 2;

	}

	if (threadIdx.x == 0){
		while (atomicCAS(mutex, 0, 1) != 0);  //lock
		if (*max < cache[0]){
			*max = cache[0];
			*maxIndex = indexCache[0];
		}
		atomicExch(mutex, 0);  //unlock
	}
}

__global__ void Modify_ImageSize_MotionCorrection(float *d_float_input, float *volume_mcorr, int new_image_Size, unsigned short top)
{
	//printf("idx = %d\n", top);
	int range = 975;
	int height = 975;
	int offset_value;
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx<487500)
	{
		int mapCmpIdx = (int)(idx / 975) * 1085 + idx % 975 + (top);
		offset_value = mapCmpIdx - idx;
		volume_mcorr[idx + offset_value] = d_float_input[idx];
		//printf("idx = %d idx + offset_value =%d\n", idx,idx + offset_value);
	}

}
__global__ void Crop_Process_Level1(float* volume_mcorr, float *volume_mcorr_crop, int new_image_size, int crop)
{

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < 487500)
	{

		int mapCmpIdx = (int)(idx / 975) * 1085 + idx % 975 + crop;
		volume_mcorr_crop[idx] = volume_mcorr[mapCmpIdx];
		//printf("idx = %d mapCmpIdx = %d  %f\n", idx, mapCmpIdx, volume_mcorr[mapCmpIdx]);
	}

}

__global__ void Crop_Process_Level2(float* volume_mcorr_crop, float *volume_mcorr_crop2, int new_image_size, int crop)
{

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < 433500)
	{
		int mapCmpIdx = (int)(idx / 867) * 975 + idx % 867 + crop;
		volume_mcorr_crop2[idx] = volume_mcorr_crop[mapCmpIdx];
		//printf("idx = %d mapCmpIdx = %d  %f\n", idx, mapCmpIdx, volume_mcorr_crop2[mapCmpIdx]);
	}

}
//*********************************************************************************************************************************//
//FFT Based Registration
__global__ void copyToComplex(float *input, Complex1 *output)
{

	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	output[idx].x = input[idx];
	output[idx].y = 0.0f;
	//if (idx<300)
	//printf("%f + i%f\n", output[idx].x, output[idx].y);
}


__global__ void fftshift2D(Complex1 *input, Complex1 *output, int width, int height)
{
	int frameSize = width*height;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int zIdx = idx / (frameSize); //0-9
	int inFrameIdx = idx % (frameSize);
	int x1 = inFrameIdx / width;
	int y1 = inFrameIdx%width;
	int outIdx = ((y1 + width / 2) % width) + ((x1 + height / 2) % height)*width + zIdx*frameSize;
	output[outIdx] = input[idx];
	//printf("%f + i%f\n", output[outIdx].x, output[outIdx].y);
}

__global__ void complexMulConj(Complex1 *Src, Complex1 *Dst, int frameNum, int frames, int width, int height, int subPixelFactor)
{
	int frameSize = width*height;
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int zIdx = idx / (frameSize);
	int inFrameIdx = idx % (frameSize);

	Complex1 temp;
	int a = (zIdx*frames + frameNum)*frameSize;
	int b = (zIdx*frames + frameNum + 1)*frameSize;

	temp.x = Src[a + inFrameIdx].x * Src[b + inFrameIdx].x - Src[a + inFrameIdx].y * (-1)*Src[b + inFrameIdx].y;
	temp.y = Src[a + inFrameIdx].x * Src[b + inFrameIdx].y*(-1) + Src[a + inFrameIdx].y * Src[b + inFrameIdx].x;

	//printf("%f + i%f\n", temp.x, temp.y);

	int outFrameIdx = 0;
	if (subPixelFactor == 1)
		outFrameIdx = idx;
	else
		outFrameIdx = (inFrameIdx / width + height / 2)* width*subPixelFactor + (inFrameIdx%width + width / 2) + zIdx*frameSize * 4;
	Dst[outFrameIdx] = temp;
	//printf("%f + i%f\n", Dst[outFrameIdx].x, Dst[outFrameIdx].y);

}


__global__ void normData(Complex1 *input, float norm)
{

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	input[idx].x *= norm;
	input[idx].y *= norm;
	//if (idx < 10)
	//printf("%f + i%f\n", input[idx].x, input[idx].y);
}
__device__ float complexAbs(Complex1 input)
{
	float output;
	output = sqrt(pow(input.x, 2) + pow(input.y, 2));
	return output;
}

__global__ void batchComplexAbs(Complex1 *Src, float *Dst, int offset)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	Dst[offset + idx] = complexAbs(Src[idx]);
	//if (idx < 10)
	//printf("%f\n", Dst[idx]);
}

__device__ void MaxWarpReduce(volatile float *sdata, unsigned int tid, int blockSize, int* loc)
{
	if (blockSize >= 64) {
		if (sdata[tid] < sdata[tid + 32]){
			sdata[tid] = sdata[tid + 32];
			loc[tid] = loc[tid + 32];
		}
	}
	if (blockSize >= 32) {
		if (sdata[tid] < sdata[tid + 16]){
			sdata[tid] = sdata[tid + 16];
			loc[tid] = loc[tid + 16];
		}
	}
	if (blockSize >= 16) {
		if (sdata[tid] < sdata[tid + 8]){
			sdata[tid] = sdata[tid + 8];
			loc[tid] = loc[tid + 8];
		}
	}
	if (blockSize >= 8){
		if (sdata[tid] < sdata[tid + 4]){
			sdata[tid] = sdata[tid + 4];
			loc[tid] = loc[tid + 4];
		}
	}
	if (blockSize >= 4) {
		if (sdata[tid] < sdata[tid + 2]){
			sdata[tid] = sdata[tid + 2];
			loc[tid] = loc[tid + 2];
		}
	}
	if (blockSize >= 2) {
		if (sdata[tid] < sdata[tid + 1]){
			sdata[tid] = sdata[tid + 1];
			loc[tid] = loc[tid + 1];
		}
	}
}



__global__ void maxReductionBatch(float *g_idata, float *maxV, unsigned int width, int height, int* loc)
{
	//The declaration for 1024 elements is arbitrary
	//As long as this is larger than blockSize, it is fine
	__shared__ float sdata[1024];
	__shared__ int sloc[1024];
	int blockSize = blockDim.x;

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;

	sdata[tid] = 0;

	for (int j = tid; j<width; j += blockSize){
		if (sdata[tid] < g_idata[(blockIdx.x)*width + j]){
			sdata[tid] = g_idata[(blockIdx.x)*width + j];
			sloc[tid] = j;
		}
	}
	__syncthreads();

	if (blockSize >= 128) {
		if (tid <  64) {
			if (sdata[tid] < sdata[tid + 64]){
				sdata[tid] = sdata[tid + 64];
				sloc[tid] = sloc[tid + 64];
			}
		}

		__syncthreads();

		if (tid < 32)
			MaxWarpReduce(sdata, tid, blockSize, sloc);

		if (tid == 0) {
			maxV[blockIdx.x] = sdata[tid];
			loc[blockIdx.x] = sloc[tid];
		}
	}

}

__global__ void computeShift(float *RegMaxV, int *RegLoc, int width,
	int height, int offsetFrame, int framesPerBuffer, float *MaxV, float *diffphase, Complex1 *data, int *shift, int subPixelFactor)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int zIdx = idx*height;
	MaxV[idx] = RegMaxV[zIdx];

	int hloc;
	int wloc;

	hloc = 0;
	wloc = RegLoc[zIdx];

	for (int j = 1; j<height; j++){
		if (MaxV[idx] < RegMaxV[zIdx + j]){
			MaxV[idx] = RegMaxV[zIdx + j];
			hloc = j;
			wloc = RegLoc[zIdx + j];
		}
	}

	int md2 = width / 2;
	int nd2 = height / 2;

	if (wloc > md2)
		shift[idx] = wloc - width + 1;
	else
		shift[idx] = wloc;
	if (hloc > nd2)
		shift[idx + framesPerBuffer / 3] = hloc - height + 1;
	else
		shift[idx + framesPerBuffer / 3] = hloc;
	shift[idx] /= subPixelFactor;
	shift[idx + framesPerBuffer / 3] /= subPixelFactor;
	// diffphase ---> Global phase difference between the two images (should be zero if images are non-negative).
	//diffphase[idx] = atan2(data[(idx*3 + offsetFrame)*width/subPixelFactor*height/subPixelFactor+ hloc/subPixelFactor*width/subPixelFactor +wloc/subPixelFactor].y,data[(idx*3 + offsetFrame)*width/subPixelFactor*height/subPixelFactor+hloc/subPixelFactor*width/subPixelFactor +wloc/subPixelFactor].x);	
	// For our OCT processing pipeline, the intensity of processed images are only from 0-1.
	diffphase[idx] = 0;

}


__global__ void getMeshgrid(int *Nw, int *Nh, int width, int height)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int dstIdx1 = int(idx / width);
	int dstIdx2 = idx%width;
	if (dstIdx2 < (width / 2))
		Nw[idx] = dstIdx2;
	else
		Nw[idx] = dstIdx2 - width;

	if (dstIdx1 < (height / 2))
		Nh[idx] = dstIdx1;
	else
		Nh[idx] = dstIdx1 - height;

	//printf("%d\n", Nh[idx]);
	//printf("%d\n", Nw[idx]);
}

__device__ Complex1 ComplexMul(Complex1 srcA, Complex1 srcB)
{
	Complex1 output;
	output.x = srcA.x * srcB.x - srcA.y * srcB.y;
	output.y = srcA.x * srcB.y + srcA.y * srcB.x;
	return output;
}


__global__ void ImagExpB(Complex1 *Src, int *Nw, int *Nh, int width, int height, int frameNum, int framesPerBuffer, int *shift, float *diffphase)

{
	float _PI =  3.14159265358979;
	float theta;
	Complex1 r;
	Complex1 s;
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int zIdx = idx / (width*height);
	int InframeIdx = idx % (width*height);
	theta = 2 * _PI*((-1)*(float(shift[zIdx])*float(Nw[InframeIdx]) / width + float(shift[zIdx + framesPerBuffer / 3])*float(Nh[InframeIdx]) / height));

	r.x = cosf(theta);
	r.y = sinf(theta);
	s.x = cosf(diffphase[zIdx]);
	s.y = sinf(diffphase[zIdx]);
	Src[(zIdx * 3 + frameNum)*width*height + InframeIdx] = ComplexMul(Src[(zIdx * 3 + frameNum)*width*height + InframeIdx], ComplexMul(r, s));
	//printf("%2f + i%2f\n", Src[(zIdx * 3 + frameNum)*width*height + InframeIdx]);
}

__global__ void Variance(float* src_Buffer, float *dst_Buffer, int numF, int frameNum, int frameSize)

{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int zIdx = idx / (frameSize); //0-9
	int inFrameIdx = idx % (frameSize);


	float tempVal;
	tempVal = 0;

	for (int i = 0; i < numF; i++)
		tempVal += src_Buffer[(zIdx * numF + frameNum + i) * frameSize + inFrameIdx]; //zIdx*numF = 0:3:6:9:12...:27


	float mean = tempVal / numF;


	float var = 0;
	for (int i = 0; i < numF; i++)
		var += (pow(src_Buffer[(zIdx * numF + frameNum + i) * frameSize + inFrameIdx] - mean, 2));

	tempVal = var / numF * 100; //The scaling factor 20 here was chosen for display purpose

	dst_Buffer[(zIdx * numF + frameNum) * frameSize + inFrameIdx] = tempVal;


}

__global__ void ZeroPad_Process(float *Axial_Motion_CorrResult, float *zero_Pad_Result, int imageSize)
{
	int tid = threadIdx.x + blockDim.x *blockIdx.x;
	if (tid < 432633)
	{
		zero_Pad_Result[tid] = Axial_Motion_CorrResult[tid];
		//printf("%2f\n", zero_Pad_Result[tid]);
	}
	else
	{
		zero_Pad_Result[tid] = 0.0;
	}
}