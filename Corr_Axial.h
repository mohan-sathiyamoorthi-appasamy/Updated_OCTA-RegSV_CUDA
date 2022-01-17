#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include<iostream>
#include<cmath>
#include "fstream"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "time.h"
#include <device_functions.h>
#include<cufft.h>
#include<conio.h>
#include<direct.h>

//#include <ctime>
using namespace cv;
//Compile kernel code for Compute 2.0 and above only
using namespace std;

typedef float2 Complex1;





//**************************************************************************************************************************************//
// Correlation based Axial Motion Correction
__global__ void floatDataTypeConversion(unsigned short *ushortData, float *Result, int hostfileLen);
__global__ void block_sum(float *input, float *per_block_results, const size_t n);
__global__ void Subtraction(float *d_result,float *meanSubFirstImage,float device_result, int host_fileLen1,int numElements);
__global__ void Square(float *Square_Result_FirstImage, float *meanSubFirstImage, int host_fileLen1);
__global__ void Mul_Operation(float *mulMeanSubImage, float * meanSubImage1, float* meanSubImage2, int hostfileLen2);
__global__ void ImageSizeModification(float *d_float_input_sizeModification, float *d_result, int imageSize);
__global__ void cropMLS(float *complexArray, float *floatArray, int width, int height, int fftWidth, int frameIdx, int offset, int range);
__global__ void absKernel(float *d_float_input, float *d_absLogData, int hostfileLen);
__global__ void logKernel(float *d_absInputData, float *d_LogInputData, int hostfileLen);
__global__ void Modify_ImageSize_MotionCorrection(float *d_float_input, float *volume_mcorr, int new_Image_Size, double top);
__global__ void find_maximum_kernel(float *array, float *max, int* maxIndex, int *mutex, unsigned int n);
__global__ void Modify_ImageSize_MotionCorrection(float *d_float_input, float *volume_mcorr, int new_image_Size, unsigned short top);
__global__ void floatTypeConversionProcess(unsigned short *ushortData, float *Result, int hostfileLen);
__global__ void Crop_Process_Level1(float* volume_mcorr, float *volume_mcorr_crop, int new_image_size, int crop);
__global__ void Crop_Process_Level2(float* volume_mcorr_crop, float *volume_mcorr_crop2, int new_image_size, int crop);
__global__ void ImagExpB(Complex1 *Src, int *Nw, int *Nh, int width, int height, int frameNum, int framesPerBuffer, int *shift, float *diffphase);

void uShortToFloatConversion(unsigned short *d_volumeBuffer, float *d_result, int host_fileLen, cudaStream_t stream);
void SizeModification(float *d_float_input_sizeModification, float *d_result, int numelements,int imageSize,cudaStream_t stream);
void Mean_Subtraction(float *meanSubFirstImage, float *input_Image1, int imageSize, int numElements);
float corrcoeff_Calculation(float *input_Image1, float *input_Image2, int imageSize, int numElements,cudaStream_t stream);
void Data_subtraction(float *d_result, float *meanSubFirstImage, float device_result, int imageSize,int numElements);
void Squared_Image(float host_SquaredFirstImage, float *host_meanSubFirstImage, int imageSize);
float Sum_calculation(float *d_result, float *mean_result, int host_fileLen);
float Block_SumCalculation(float *host_SquaredFirstImage, int imageSize);
void multiplication(float *mulMeanSubImage, float * meanSubImage1, float* meanSubImage2, int hostfileLen2);
void postFFTCrop(float *d_ComplexArray, float *dev_processBuffer, int frames, int frameIdx, int offset, int range,cudaStream_t stream);
void absLogProcess(float *d_absInputData, float *d_LogInputData, int host_fileLen);
void abs_CorrResult(float *corr_result_stack, float* abs_corr_result_stack, int fileLen);
int Max_Location(float *corr_result_stack);
bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);
float IndexMaxElement(float *corr_result_stack, int Total_Offset);
float FindMaxValue(float* disp_Ind, int numFrames);
float FindAbsMinValue(float* disp_Ind, int numFrames);
void Image_Assign_NewSize(float *d_float_input_copy, float *volume_mcorr, int new_image_Size, unsigned short top);
//void corrcoeff_Calculation(float *input_Image1, float *input_Image2, int imageSize, int numElements,float* corr);
void floatConversionProcess(unsigned short *d_volumeBuffer, float *d_result_copy, int host_fileLen);
void crop_level1(float *volume_mcorr, float *volume_mcorr_crop, int new_image_Size, int crop);
void crop_level2(float *volume_mcorr_crop, float *volume_mcorr_crop2, int new_image_Size, int crop);
//******************************************************************************************************************************************************//
void SaveTifImage1(Mat ImgVector, char *fileNameImage, int imageSize, int i);
//Mat getMean(const vector<Mat>&images);
Mat logTypeConversion(float* temp_RegImage);
void writeMatToFile(cv::Mat m, const char* filename);
Mat map(Mat x, float in_min, float in_max, long out_min, long out_max);
Mat getMean(const vector<Mat>&images, char*RegDirName, int imageSize, int ctr);
//*****************************************************************************************************************************************************//
//FFT Based Registration

void dftregistration(float *Src, int subPixelFactor, int width, int height, int numF, int frameNum);
void getMeshgridFunc(int width, int height);
void regisMulB(Complex1 *src, int *Nw, int *Nh, int width, int height, int frameNum, int framesPerBufferm, float *diffphase, int *shift);


__global__ void copyToComplex(float *input, Complex1 *output);
__global__ void fftshift2D(Complex1 *input, Complex1 *output, int width, int height);
__global__ void Complex1MulConj(Complex1 *Src, Complex1 *Dst, int frameNum, int frames, int width, int height, int subPixelFactor);

__global__ void normData(Complex1 *input, float norm);
__global__ void batchComplexAbs(Complex1 *Src, float *Dst, int offset);
__device__ float complexAbs(Complex1 input);
__global__ void maxReductionBatch(float *g_idata, float *maxV, unsigned int width, int height, int* loc);
__device__ void MaxWarpReduce(volatile float *sdata, unsigned int tid, int blockSize, int* loc);
__global__ void computeShift(float *RegMaxV, int *RegLoc, int width, int height, int offsetFrame, int framesPerBuffer, float *MaxV, float *diffphase, Complex1 *data, int *shift, int subPixelFactor);
__global__ void getMeshgrid(int *Nw, int *Nh, int width, int height);
__global__ void ImagExpB(Complex1 *Src, int *Nw, int *Nh, int width, int height, int frameNum, int framesPerBuffer, int *shift, float *diffphase);
__global__ void complexMulConj(Complex1 *Src, Complex1 *Dst, int frameNum, int frames, int width, int height, int subPixelFactor);


//Speckle Variance
void speckleVar(float *d_volumeBuffer, float *specVar_result, int width, int height, int numF, int frameNum);
__global__ void Variance(float* src_Buffer, float *dst_Buffer, int numF, int frameNum, int frameSize);

void PaddingZeros(float *Axial_Motion_CorrResult, float *zero_Pad_Result, int imageSize);
__global__ void ZeroPad_Process(float *Axial_Motion_CorrResult, float *zero_Pad_Result, int imageSize);

void SaveTifImage(float *temp_RegImage, char *fileNameImage, int imageSize, int i);
//void SaveTifImage(float *temp_RegImage, Mat flipImage);
//Mat1b getMean(const vector<Mat1f>&images);
//Mat1b getMean(const vector<Mat1f>&images, char* RegDirName, int imageSize, int ctr);