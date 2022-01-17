#include "Corr_Axial.h"
cufftHandle fft2d_Batchplan;
cufftHandle fft2d_BatchplanS;


float Sum_calculation(float *d_result, float *mean_result, int host_fileLen)
{
	const size_t n = host_fileLen;
	//dim3 dimBlock(numThreadsPerBlock);
	//dim3 dimGrid(host_fileLen / dimBlock.x);

	const size_t block_size = 512;
	const size_t num_blocks = (host_fileLen / block_size) + ((host_fileLen % block_size) ? 1 : 0);

	block_sum << <num_blocks, block_size, block_size*sizeof(float) >> >(d_result, mean_result, host_fileLen);
	
	//cudaMemcpy(&device_result1, mean_result , sizeof(float), cudaMemcpyDeviceToHost);

	// launch a single block to compute the sum of the partial sums
	block_sum << <1, num_blocks, num_blocks * sizeof(float) >> >(mean_result, mean_result + num_blocks, num_blocks);

	float device_result1 = 0;
	cudaMemcpy(&device_result1, mean_result + num_blocks, sizeof(float), cudaMemcpyDeviceToHost);

	//cout << "Device Result1: " << device_result1 << endl;
	return device_result1;

}


void Data_subtraction(float *d_result, float *meanSubFirstImage, float device_result, int imageSize, int numElements)
{
	const size_t block_size = 512;
	const size_t num_blocks = (numElements / block_size) + ((numElements%block_size) ? 1 : 0);
	Subtraction << <num_blocks, block_size >> >(d_result, meanSubFirstImage, device_result, imageSize, numElements);
}

void Mean_Subtraction(float *meanSubFirstImage,float *input_Image1,int imageSize,int numElements)
{
	const size_t block_size = 512;
	const size_t num_blocks = (numElements / block_size) + ((numElements % block_size) ? 1 : 0);

	float *cumulative_Sum_result = 0;
	cudaMalloc((void**)&cumulative_Sum_result, sizeof(float) * (num_blocks + 1));

	float mean_result_1 = 0;
	mean_result_1 = Sum_calculation(input_Image1, cumulative_Sum_result, numElements);

	//std::cout << "Device Result Final: " << cumulative_Sum_result << std::endl;
	// copy the result back to the host
	float device_result = 0;
	cudaMemcpy(&device_result, cumulative_Sum_result + num_blocks, sizeof(float), cudaMemcpyDeviceToHost);

	//std::cout << "Device Result Final: " << device_result << std::endl;

	float Mean_Result_FirstImage = 0;
	Mean_Result_FirstImage = device_result / imageSize;
	//std::cout << "Device Mean: " << Mean_Result_FirstImage << std::endl;

	//float *meanSubFirstImage;
	//cudaMalloc((void**)&meanSubFirstImage, sizeof(float)*imageSize);

	//Subtraction Mean Value from First Image 
	Data_subtraction(input_Image1, meanSubFirstImage, Mean_Result_FirstImage, imageSize, numElements);
	cudaFree(cumulative_Sum_result);
	

}


void Squared_Image(float *host_SquaredFirstImage, float *host_meanSubFirstImage, int imageSize)
{
		const size_t block_size = 512;
		const size_t num_blocks = (imageSize / block_size) + ((imageSize%block_size) ? 1 : 0);
		Square << <block_size, num_blocks >> >(host_SquaredFirstImage, host_meanSubFirstImage, imageSize);
}

float Block_SumCalculation(float *host_SquaredFirstImage,int imageSize)
{
	const size_t block_size = 512;
	const size_t num_blocks = (imageSize / block_size) + ((imageSize % block_size) ? 1 : 0);

	float *mean_result = 0;
	cudaMalloc((void**)&mean_result, sizeof(float) * (num_blocks + 1));

	float mean_result_1 = 0;
	mean_result_1 = Sum_calculation(host_SquaredFirstImage, mean_result, imageSize);
	return mean_result_1;
	cudaFree(mean_result);
	
}

void multiplication(float *mulMeanSubImage, float * meanSubImage1, float* meanSubImage2, int imageSize)
{
	int numThreadsPerBlock = 256;
	while (imageSize / numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	dim3 dimBlock(numThreadsPerBlock);
	dim3 dimGrid(imageSize / dimBlock.x);
	//printf("kernel is Start");
	Mul_Operation << <dimGrid, dimBlock >> >(mulMeanSubImage, meanSubImage1, meanSubImage2, imageSize);
}


//*************************************************************************************************************//
// Process - Correlation Coefficients
void uShortToFloatConversion(unsigned short *d_volumeBuffer, float *d_result, int host_fileLen,cudaStream_t stream)
{
	//printf("%d", host_fileLen);
	int numThreadsPerBlock = 256;
	while (host_fileLen / numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	dim3 dimBlock(numThreadsPerBlock);
	int roundedValue = ceil(host_fileLen / dimBlock.x)+1;
	dim3 dimGrid(roundedValue);
	//printf("%d", roundedValue);
	floatDataTypeConversion << <dimGrid, dimBlock,0,stream >> >(d_volumeBuffer, d_result,host_fileLen);
}

void floatConversionProcess(unsigned short *d_volumeBuffer, float *d_result_copy, int host_fileLen)
{
	//printf("%d", host_fileLen);
	int numThreadsPerBlock = 256;
	while (host_fileLen / numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	dim3 dimBlock(numThreadsPerBlock);
	int roundedValue = ceil(host_fileLen / dimBlock.x) + 1;
	dim3 dimGrid(roundedValue);
	//printf("%d", roundedValue);
	floatTypeConversionProcess << <dimGrid, dimBlock >> >(d_volumeBuffer,d_result_copy, host_fileLen);
}



// 20 Log  of Input Data
void absLogProcess(float *d_absInputData, float *d_LogInputData, int host_fileLen)
{
	int numThreadsPerBlock = 256;
	while (host_fileLen / numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	dim3 dimBlock(numThreadsPerBlock);
	int roundedValue = ceil(host_fileLen / dimBlock.x) + 1;
	dim3 dimGrid(roundedValue);
	//printf("%d", roundedValue);
	logKernel << <dimGrid, dimBlock >> >(d_absInputData, d_LogInputData, host_fileLen);
}


void abs_CorrResult(float *corr_result_stack, float* abs_corr_result_stack, int fileLen)
{
	int numThreadsPerBlock = 256;
	while (fileLen / numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	dim3 dimBlock(numThreadsPerBlock);
	int roundedValue = ceil(fileLen / dimBlock.x) + 1;
	dim3 dimGrid(roundedValue);
	//printf("%d", roundedValue);
	//printf("abs kernel start");
	absKernel << <dimGrid, dimBlock >> >(corr_result_stack, abs_corr_result_stack, fileLen);
}

void SizeModification(float *d_float_input_sizeModification, float *d_result, int numelements,int imageSize,cudaStream_t stream)
{
	int numThreadsPerBlock = 256;
	while (numelements / numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	dim3 dimBlock(numThreadsPerBlock);
	//int roundedValue = ceil(numelements / dimBlock.x) + 1;
	//dim3 dimGrid(roundedValue);
	dim3 dimGrid(numelements / dimBlock.x);
	ImageSizeModification << <dimGrid, dimBlock,0,stream >> >(d_float_input_sizeModification, d_result, imageSize);
}


//The range var is a portion of the width far, eg width = 1024, a quarter of the width would be the range = 256
void postFFTCrop(float *d_ComplexArray, float *dev_processBuffer, int frames, int frameIdx, int offset, int range,cudaStream_t stream)
{
	
	int numThreadsPerBlock = 256;
	int frameHeight = 975;
	int fftLengthMult = 1;
	int frameWidth = 500;


	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(range*frameHeight*frames / dimBlockX.x);


	//printf("%d\n", range*frameHeight*frames);
	//MLS = Modulus, Log, and Scaling
	//This method of post FFT crops out a certain portion of the data, and copies into buffer
	//This method conserves resolution, but cuts down on the viewing range
	//As opposed to the other method which downsizes the whole volume
	 
	 
	cropMLS << <dimGridX, dimBlockX,0,stream>> >(d_ComplexArray,dev_processBuffer,frameWidth, frameHeight,fftLengthMult*frameWidth, frameIdx, offset, range);

}


float corrcoeff_Calculation(float *input_Image1, float *input_Image2, int imageSize,int numElements,cudaStream_t stream)
{
	
	//Section:1 Find the CorrCoeff Parameters values for Image1
	//Mean Subtraction
	float *host_meanSubFirstImage = NULL;
	cudaMalloc((void**)&host_meanSubFirstImage, sizeof(float)* numElements);


	//***************************//
	Mean_Subtraction(host_meanSubFirstImage, input_Image1, imageSize,numElements);
	//***************************//
	
	//Squared Image
	float *host_SquaredFirstImage = NULL;
	cudaMalloc((void**)&host_SquaredFirstImage, sizeof(float)*numElements);

	//************************//
	Squared_Image(host_SquaredFirstImage, host_meanSubFirstImage, numElements);
	//************************//

	

	//Sum of Squared Image
	//************************//
	float block_Sum_Result_FirstImage;
	block_Sum_Result_FirstImage = Block_SumCalculation(host_SquaredFirstImage, numElements);
	//std::cout << "Result sum squared second image: " << block_Sum_Result_FirstImage << std::endl;
	//************************//

	//*******************************************************************************************//

	//Section:2 Find the CorrCoeff Parameters values for Image2
	//Mean Subtraction
	float *host_meanSubFirstImage2 = NULL;
	cudaMalloc((void**)&host_meanSubFirstImage2, sizeof(float)*numElements);


	//***************************//
	Mean_Subtraction(host_meanSubFirstImage2, input_Image2, imageSize, numElements);
	//***************************//
	
	
	//Squared Image
	float *host_SquaredFirstImage2 = NULL;
	cudaMalloc((void**)&host_SquaredFirstImage2, sizeof(float)*numElements);

	//************************//
	Squared_Image(host_SquaredFirstImage2, host_meanSubFirstImage2, numElements);
	//************************//
	

	//Sum of Squared Image
	//************************//
	float block_Sum_Result_FirstImage2;
	block_Sum_Result_FirstImage2 = Block_SumCalculation(host_SquaredFirstImage2, numElements);
	//std::cout << "Result sum squared second image: " << block_Sum_Result_FirstImage2 << std::endl;
	//************************//
	
	
	//*****************************************************************************************************//
	//Multiplication of Subtracted Mean of First Image and Second Image
	float *mulMeanSubImage = NULL;
	cudaMalloc((void**)&mulMeanSubImage, numElements * sizeof(float));
	multiplication(mulMeanSubImage, host_meanSubFirstImage, host_meanSubFirstImage2,numElements);


	/*float *test = NULL;
	test = (float*)malloc(numElements* sizeof(float));
	cudaMemcpy(test, mulMeanSubImage, numElements * sizeof(float), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < 50; i++)
	{
		printf("%2f\n", test[i]);
	}
*/

	//***************************************************************************************************//

	//Numerator CC - Sum of Multiplication of Subtracted Mean of First Image and Second Image
	float numerator_CC;
	numerator_CC = Block_SumCalculation(mulMeanSubImage, numElements);
	

	//Denominator CC - Multiplication of Sum of Squared mean Sub I & II Image
	float denominator_CC;
	denominator_CC = sqrt(block_Sum_Result_FirstImage * block_Sum_Result_FirstImage2);
	//std::cout << "Result sum squared second image: " << denominator_CC << std::endl;

	//**************************************************************************************************//  
	
	float corr_result = numerator_CC /denominator_CC;
	//std::cout << "Result sum squared second image: " << corr_result << std::endl;






	cudaFree(host_meanSubFirstImage);
	cudaFree(host_meanSubFirstImage2);
	cudaFree(host_SquaredFirstImage);
	cudaFree(host_SquaredFirstImage2);
	cudaFree(mulMeanSubImage);
	 

	
	//Test
	
	return corr_result;
}


void Image_Assign_NewSize(float *d_float_input_copy, float *volume_mcorr,int new_image_Size, double top)
{
	int numThreadsPerBlock = 256;
	while (new_image_Size / numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	dim3 dimBlock(numThreadsPerBlock);
	int roundedValue = ceil(new_image_Size / dimBlock.x) + 1;
	dim3 dimGrid(roundedValue);
	//printf("%d\n", roundedValue);
	Modify_ImageSize_MotionCorrection << <dimGrid, dimBlock >> >(d_float_input_copy, volume_mcorr, new_image_Size, top);
}


float IndexMaxElement(float* corr_result_stack,int Total_Offset)
{
	float *maximum = NULL;
	maximum = (float*)malloc(41 * sizeof(float));
	memcpy(maximum, corr_result_stack, 41 * sizeof(float));

	float location = 0.0;
	for (int c = 1; c < (Total_Offset + 1); c++)
	{
		if (*(corr_result_stack + c) > *maximum)
		{
			*maximum = *(corr_result_stack + c);
			location = c+1;
		}

	}
	return location;
	free(maximum);
}


bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
	//Number of key points
	int N = key_point.size();

	//Construct matrix X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) + std::pow(key_point[k].x, i + j);
			}
		}
	}

	//Construct matrix Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) + std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}

	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//Solve matrix A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}

 

float FindMaxValue(float* disp_Ind, int numFrames)
{
	float *maximum = NULL;
	maximum = (float*)malloc(numFrames * sizeof(float));
	memcpy(maximum, disp_Ind, numFrames * sizeof(float));

	float location = 0.0;
	for (int c = 1; c < (numFrames + 1); c++)
	{
		if (*(disp_Ind + c) > *maximum)
		{
			*maximum = *(disp_Ind + c);
			location = c + 1;
		}

	}
	return *maximum;
	free(maximum);
}


float FindAbsMinValue(float* disp_Ind, int numFrames)
{
	float *minimum = NULL;
	minimum = (float*)malloc(numFrames * sizeof(float));
	memcpy(minimum, disp_Ind, numFrames * sizeof(float));

	float location = 0.0;
	for (int c = 1; c < (numFrames + 1); c++)
	{
		if (*(disp_Ind + c) < *minimum)
		{
			*minimum = *(disp_Ind + c);
			location = c + 1;
		}

	}
	return abs(*minimum);
	free(minimum);
}

void Image_Assign_NewSize(float *d_float_input_copy, float *volume_mcorr, int new_image_Size, unsigned short top)
{

	int numThreadsPerBlock = 256;
	while (new_image_Size / numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	dim3 dimBlock(numThreadsPerBlock);
	int roundedValue = ceil(new_image_Size / dimBlock.x) + 1;
	dim3 dimGrid(roundedValue);
	//printf("%u\n",top);
	Modify_ImageSize_MotionCorrection << <dimGrid, dimBlock >> >(d_float_input_copy, volume_mcorr, new_image_Size, top);
}


void crop_level1(float *volume_mcorr, float *volume_mcorr_crop, int new_image_Size, int crop)
{
	int numThreadsPerBlock = 256;
	while (new_image_Size / numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	dim3 dimBlock(numThreadsPerBlock);
	int roundedValue = ceil(new_image_Size / dimBlock.x) + 1;
	dim3 dimGrid(roundedValue);
	//printf("%u\n",top);
	Crop_Process_Level1 << <dimGrid, dimBlock >> >(volume_mcorr, volume_mcorr_crop, new_image_Size, crop);
}



void crop_level2(float *volume_mcorr_crop, float *volume_mcorr_crop2, int new_image_Size, int crop)
{
	int numThreadsPerBlock = 256;
	while (new_image_Size / numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	dim3 dimBlock(numThreadsPerBlock);
	int roundedValue = ceil(new_image_Size / dimBlock.x) + 1;
	dim3 dimGrid(roundedValue);
	//printf("%u\n",top);
	Crop_Process_Level2 << <dimGrid, dimBlock >> >(volume_mcorr_crop, volume_mcorr_crop2, new_image_Size, crop);
}



//**************************************************************************************************************************//
//FFT Based Registration
int *Nw;
int *Nh;

void getMeshgridFunc(int width, int height)
{

	int numThreadsPerBlock = 256;
	cudaMalloc((void**)&Nw, width*height*sizeof(int));
	cudaMalloc((void**)&Nh, width*height*sizeof(int));
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(width*height / dimBlockX.x);
	getMeshgrid << <dimGridX, dimBlockX >> >(Nw, Nh, width, height);
}


void regisMulB(Complex1 *src, int *Nw, int *Nh, int width, int height, int frameNum, int framesPerBufferm, float *diffphase, int *shift){
	 
	int numThreadsPerBlock = 256;
	int framesPerBuffer = 3;
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(width*height*framesPerBuffer / 3 / dimBlockX.x);
	ImagExpB << <dimGridX, dimBlockX >> >(src, Nw, Nh, width, height, frameNum, framesPerBuffer, shift, diffphase);
}



void dftregistration(float *Src,
	int subPixelFactor,
	int width,
	int height,
	int numF,
	int frameNum)
{

	int numThreadsPerBlock = 256;
	int framesPerBuffer = 3;
	unsigned int bufferSize;
	bufferSize = width*height*framesPerBuffer;



	while ((bufferSize / numThreadsPerBlock) > 65535)

	{

		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock > 1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}

	//**********1. Create Input Data as Complex Data*************************************************************************************
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(width*height*framesPerBuffer / dimBlockX.x);


	Complex1 *sub_FFTCompBufferBatch;
	cudaMalloc((void**)&sub_FFTCompBufferBatch, width*height*framesPerBuffer *sizeof(Complex1));

	copyToComplex << <dimGridX, dimBlockX >> > (Src, sub_FFTCompBufferBatch);



	//******** 2.sub-pixel registration****************************************************************************************************

	// subPixelFactor ---> Upsampling factor (integer)
	// subPixelFactor = 1 ----> whole-pixel shift -- Compute crosscorrelation by an IFFT and locate the peak
	// subPixelFactor = 2 ----> Images will be registered to within 1/2 of a pixel.
	Complex1 *sub_FFTCompBufferBatchSmallTemp;

	dim3 dimGridXS(width*height*framesPerBuffer / dimBlockX.x / 3);


	// Update the width and height
	int Nwidth = width*subPixelFactor;
	int Nheight = height*subPixelFactor;
	dim3 dimGridXfac(Nwidth*Nheight*framesPerBuffer / dimBlockX.x / 3);


	cudaMalloc((void**)&sub_FFTCompBufferBatchSmallTemp, subPixelFactor*subPixelFactor*width*height*framesPerBuffer / 3 * sizeof(Complex1));
	cudaMemset(sub_FFTCompBufferBatchSmallTemp, 1, subPixelFactor*subPixelFactor*width*height *framesPerBuffer / 3 * sizeof(Complex1));


	Complex1 *sub_FFTCompBufferBatchSmall;
	cudaMalloc((void**)&sub_FFTCompBufferBatchSmall, subPixelFactor*subPixelFactor*width*height*framesPerBuffer / 3 * sizeof(Complex1));
	cudaMemset(sub_FFTCompBufferBatchSmall, 0, subPixelFactor*subPixelFactor*width*height *framesPerBuffer / 3 * sizeof(Complex1));

	Complex1 *sub_FFTCompBufferBatchTemp;
	cudaMalloc((void**)&sub_FFTCompBufferBatchTemp, width*height*framesPerBuffer *sizeof(Complex1));

	//***************************************************************************************************************//
	// Section 1: To correct the 1st and 2nd B-scans in a BM-scan

	if (subPixelFactor >1)
	{
		// subPixelFactor = 2 ----> First upsample by a factor of 2 to obtain initial estimate
		// Embed Fourier data in a 2x larger array
		cufftExecC2C(fft2d_Batchplan,
			(cufftComplex *)sub_FFTCompBufferBatch,
			(cufftComplex *)sub_FFTCompBufferBatch,
			CUFFT_FORWARD);

		fftshift2D << <dimGridX, dimBlockX >> >
			(sub_FFTCompBufferBatch, sub_FFTCompBufferBatchTemp, width, height);// Batch Image Size(3*w*h) - Output

		complexMulConj << <dimGridXS, dimBlockX >> >
			(sub_FFTCompBufferBatchTemp, sub_FFTCompBufferBatchSmall, 0, numF, width, height, subPixelFactor);//singleImageSize*twice(upsampfactor)-output

		fftshift2D << <dimGridXfac, dimBlockX >> >
			(sub_FFTCompBufferBatchSmall, sub_FFTCompBufferBatchSmallTemp, Nwidth, Nheight);//singleImageSize*twice(upsampfactor)-output

		cufftExecC2C(fft2d_BatchplanS,
			(cufftComplex *)sub_FFTCompBufferBatchSmallTemp,
			(cufftComplex *)sub_FFTCompBufferBatchSmallTemp,
			CUFFT_INVERSE);
	}
	else
	{
		cufftExecC2C(fft2d_Batchplan,
			(cufftComplex *)sub_FFTCompBufferBatch,
			(cufftComplex *)sub_FFTCompBufferBatch,
			CUFFT_FORWARD);

		complexMulConj << <dimGridXS, dimBlockX >> >
			(sub_FFTCompBufferBatch, sub_FFTCompBufferBatchSmallTemp, 0, numF, width, height, subPixelFactor);

		cufftExecC2C(fft2d_BatchplanS,
			(cufftComplex *)sub_FFTCompBufferBatchSmallTemp,
			(cufftComplex *)sub_FFTCompBufferBatchSmallTemp,
			CUFFT_INVERSE);
	}

	//***************3.Normalization**************************************************************************
	normData << <dimGridXfac, dimBlockX >> >(sub_FFTCompBufferBatchSmallTemp, 1 / float(Nwidth*Nheight));

	//***************4.Absolute Value of Data*****************************************************************
	float *sub_absBatchSmall;
	cudaMalloc((void**)&sub_absBatchSmall, subPixelFactor*subPixelFactor*width*height*framesPerBuffer / 3 * sizeof(float));
	batchComplexAbs << <dimGridXfac, dimBlockX >> >(sub_FFTCompBufferBatchSmallTemp, sub_absBatchSmall, 0);

	//**************5.Compute Cross Correlation & Locate the Peak*********************************************
	float *RegMaxVB;
	int *RegLocB;
	const int blockSize = 128;
	dim3 dimGridX1(Nheight*framesPerBuffer / 3);
	dim3 dimBlockX1(blockSize);

	cudaMalloc((void**)&RegMaxVB, framesPerBuffer / 3 * height*subPixelFactor *sizeof(float));
	cudaMalloc((void**)&RegLocB, framesPerBuffer / 3 * height*subPixelFactor *sizeof(int));
	maxReductionBatch << <dimGridX1, dimBlockX1 >> >(sub_absBatchSmall, RegMaxVB, Nwidth, Nheight, RegLocB);

	//*************6.Find Shift in Original Pixel Grid from the Position of Cross Correlation Peak*************
	float *MaxVB;
	int *shiftB;
	float *diffphaseB;

	cudaMalloc((void**)&MaxVB, framesPerBuffer / 3 * sizeof(float));
	cudaMalloc((void**)&shiftB, framesPerBuffer / 3 * 2 * sizeof(int));
	cudaMalloc((void**)&diffphaseB, framesPerBuffer / 3 * sizeof(float));

	if (subPixelFactor >1)
		computeShift << <1, framesPerBuffer / 3 >> >(RegMaxVB, RegLocB, Nwidth, Nheight, 0, framesPerBuffer, MaxVB, diffphaseB, sub_FFTCompBufferBatchSmallTemp, shiftB, subPixelFactor);
	else
		computeShift << <1, framesPerBuffer / 3 >> >(RegMaxVB, RegLocB, Nwidth, Nheight, 0, framesPerBuffer, MaxVB, diffphaseB, sub_FFTCompBufferBatch, shiftB, subPixelFactor);

	//***********7.Creation Mesh Grid **************************************************************************
	getMeshgridFunc(width, height);

	//**********8.Find Angle and Complex Multiplication*********************************************************
	regisMulB(sub_FFTCompBufferBatch, Nw, Nh, width, height, 1, framesPerBuffer, diffphaseB, shiftB);

	//***********************************************************************************************************
	//Section 2: To correct the the 3rd scan based on the corrected 2nd B-scans in a BM-scan
	//printf("Registration 2 & 3 B- Scan");
	if (subPixelFactor>1)
	{
		fftshift2D << <dimGridX, dimBlockX >> >
			(sub_FFTCompBufferBatch, sub_FFTCompBufferBatchTemp, width, height);

		complexMulConj << <dimGridXS, dimBlockX >> >
			(sub_FFTCompBufferBatchTemp, sub_FFTCompBufferBatchSmall, 1, numF, width, height, subPixelFactor);

		fftshift2D << <dimGridXfac, dimBlockX >> >
			(sub_FFTCompBufferBatchSmall, sub_FFTCompBufferBatchSmallTemp, Nwidth, Nheight);
		cufftExecC2C(fft2d_BatchplanS,
			(cufftComplex *)sub_FFTCompBufferBatchSmallTemp,
			(cufftComplex *)sub_FFTCompBufferBatchSmallTemp,
			CUFFT_INVERSE);
	}
	else
	{
		complexMulConj << <dimGridXS, dimBlockX >> >
			(sub_FFTCompBufferBatch, sub_FFTCompBufferBatchSmallTemp, 1, numF, width, height, subPixelFactor);
		cufftExecC2C(fft2d_BatchplanS,
			(cufftComplex *)sub_FFTCompBufferBatchSmallTemp,
			(cufftComplex *)sub_FFTCompBufferBatchSmallTemp,
			CUFFT_INVERSE);
	}

	normData << <dimGridXfac, dimBlockX >> >(sub_FFTCompBufferBatchSmallTemp, 1 / float(Nwidth*Nheight));

	batchComplexAbs << <dimGridXfac, dimBlockX >> >(sub_FFTCompBufferBatchSmallTemp, sub_absBatchSmall, 0);

	// Compute crosscorrelation and locate the peak 
	maxReductionBatch << <dimGridX1, dimBlockX1 >> >
		(sub_absBatchSmall, RegMaxVB, Nwidth, Nheight, RegLocB);
	//Obtain shift in original pixel grid from the position of the crosscorrelation peak 
	if (subPixelFactor>1)
		computeShift << <1, framesPerBuffer / 3 >> >
		(RegMaxVB, RegLocB, Nwidth, Nheight, 1, framesPerBuffer, MaxVB, diffphaseB, sub_FFTCompBufferBatchSmallTemp, shiftB, subPixelFactor);
	else
		computeShift << <1, framesPerBuffer / 3 >> >
		(RegMaxVB, RegLocB, Nwidth, Nheight, 1, framesPerBuffer, MaxVB, diffphaseB, sub_FFTCompBufferBatch, shiftB, subPixelFactor);

	regisMulB(sub_FFTCompBufferBatch, Nw, Nh, width, height, 2, framesPerBuffer, diffphaseB, shiftB);

	cufftExecC2C(fft2d_Batchplan,
		(cufftComplex *)sub_FFTCompBufferBatch,
		(cufftComplex *)sub_FFTCompBufferBatch,
		CUFFT_INVERSE);
	normData << <dimGridX, dimBlockX >> >(sub_FFTCompBufferBatch, 1 / float(width*height));
	batchComplexAbs << <dimGridX, dimBlockX >> >(sub_FFTCompBufferBatch, Src, 0);

	cudaFree(RegMaxVB);
	cudaFree(RegLocB);
	cudaFree(sub_absBatchSmall);
	cudaFree(sub_FFTCompBufferBatchSmall);
	cudaFree(sub_FFTCompBufferBatchTemp);
	cudaFree(sub_FFTCompBufferBatchSmallTemp);
	cudaFree(sub_FFTCompBufferBatch);
}

//Speckle Variance 
void speckleVar(float* d_result, float *specVar_result, int width, int height, int numF, int frameNum)
{
	if (d_result == NULL) {
		//d_volumeBuffer = dev_tempBuffer;
		printf("Memory is Empty");
	}
	//cudaStream_t kernelStream;
	int numThreadsPerBlock = 256;
	int framesPerBuffer = 3;
	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX(width * height *framesPerBuffer / 3 / dimBlockX.x);

	//printf("numThreadsPerBlock: %d\n", width * height *framesPerBuffer / 1 / dimBlockX.x);

	Variance << <dimGridX, dimBlockX >> >(d_result, specVar_result, numF, frameNum, width * height);
}


void PaddingZeros(float *Axial_Motion_CorrResult, float *zero_Pad_Result, int imageSize)
{
	int numThreadsPerBlock = 256;
	while (imageSize / numThreadsPerBlock > 65535) {
		numThreadsPerBlock <<= 1;
		if (numThreadsPerBlock>1024) {
			printf("Error, Buffer Size is too large, Kernel is unable to Handle this kernel size!\n");
			printf("Exiting Program...");
			exit(1);
		}
	}
	dim3 dimBlock(numThreadsPerBlock);
	int roundedValue = ceil(imageSize / dimBlock.x) + 1;
	dim3 dimGrid(roundedValue);
	//printf("%u\n",top);
	ZeroPad_Process << <dimGrid, dimBlock >> >(Axial_Motion_CorrResult, zero_Pad_Result, imageSize);
}

void SaveTifImage(float *temp_RegImage,char *fileNameImage,int imageSize, int i)
//void SaveTifImage(float *temp_RegImage,Mat flipImage)
{
	double brightness = -7.0;
	double contrast = 0.9;
	double dB_Range = 50.0;
	
	Mat ImgVector(500, 867, CV_32FC1, temp_RegImage);
	
	//Find Maximum Value
	double minVal, maxVal;
	minMaxLoc(ImgVector, &minVal, &maxVal);
	
	Mat norm1;
	log(ImgVector / maxVal, norm1);
	 
	norm1 = norm1 / log(10);
	Mat norm2 = ((20 * norm1) + dB_Range) / dB_Range * 255;
	Mat norm3 = (norm2 * contrast) + brightness;

	 
	//Rotate & Flip Image
	rotate(norm3, norm3, cv::ROTATE_90_CLOCKWISE);
	Mat flipImage;
	flip(norm3, flipImage, 1);

	//Save as Tif File
	char frameNo[200];
	sprintf_s(frameNo, "%s\\%03i.tif", fileNameImage, i);
	flipImage.convertTo(flipImage, CV_8UC1);
	imwrite(frameNo, flipImage); 
}


Mat logTypeConversion(Mat ImgVector)
{
	double brightness = -7.0;
	double contrast = 0.9;
	double dB_Range = 50.0;

	//Mat ImgVector(500, 867, CV_32FC1, temp_RegImage);

	//Find Maximum Value
	double minVal, maxVal;
	minMaxLoc(ImgVector, &minVal, &maxVal);

	Mat norm1;
	log(ImgVector / maxVal, norm1);

	norm1 = norm1 / log(10);
	Mat norm2 = ((20 * norm1) + dB_Range) / dB_Range * 255;
	Mat norm3 = (norm2 * contrast) + brightness;


	rotate(norm3, norm3, cv::ROTATE_90_CLOCKWISE);

	Mat flipImage;
	flip(norm3, flipImage, 1);
	return flipImage;
}



Mat getMean(const vector<Mat>&images,char*RegDirName,int imageSize,int ctr)
{
	if (images.empty()) return Mat();
	
	Mat m(images[0].rows, images[0].cols, CV_32FC1);
	//cout << images[0].rows << endl;
	m.setTo(Scalar(0, 0, 0, 0));
	 
	Mat temp(images[0].rows, images[0].cols, CV_32FC1);
	//cout << "Get Mean: " << images.size() << endl;
	for (int i = 0; i < images.size(); ++i)
	{
		//cout << "Convert to 32FC1 " << endl;
		images[i].convertTo(temp, CV_32FC1);
		//cout << "Cummulative Add " << m.type() << ", " << temp.type() << endl;
		m +=temp;
		
	}

	double min, max;
	minMaxLoc(m, &min, &max);

	Mat M1;
	M1 = map(m, min, max,0,1);

	double brightness = -11.0;
	double contrast = 0.9;
	double dB_Range = 50.0;

	//Rotate & Flip Image
	rotate(M1, M1, cv::ROTATE_90_CLOCKWISE);
	Mat flipImage;
	flip(M1, flipImage, 1);

	Mat norm1;
	log(flipImage / max, norm1);

	norm1 = norm1 / log(10);
	Mat norm2 = ((20 * norm1) + dB_Range) / dB_Range*255;
	Mat norm3 = (norm2 * contrast) + brightness;

	//Save as Tif File
	char frameNo[200];
	sprintf_s(frameNo, "%s\\%03i.tif", RegDirName, ctr);
	norm3.convertTo(norm3, CV_8UC1,1./images.size());
	imwrite(frameNo, norm3);
	return M1;
	
}

Mat map( Mat x, float in_min, float in_max, long out_min, long out_max)
{

	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void SaveTifImage1(Mat ImgVector1, char* fileNameImage, int imageSize, int i)
//void SaveTifImage(float *temp_RegImage,Mat flipImage)
{
	//double brightness = -7.0;
	//double contrast = 0.9;
	//double dB_Range = 50.0;
	//Mat ImgVector = ImgVector1.reshape(867,500);
	//cout << ImgVector << endl;
	 
	//cout << ImgVector.type() << endl;
	//Find Maximum Value
	//double minVal, maxVal;
	//minMaxLoc(ImgVector1, &minVal, &maxVal);

	//ImgVector1.convertTo(ImgVector1,CV_32FC1);
	//Mat norm1;
	//log(ImgVector1 / maxVal, norm1);

	//
	//norm1 = norm1 / log(10);
	//
	//Mat norm2 = ((20 * norm1) + dB_Range) / dB_Range * 255;
	//Mat norm3 = (norm2 * contrast) + brightness;
	////cout << norm3 << endl;

	//rotate(norm3, norm3, cv::ROTATE_90_CLOCKWISE);

	//Mat flipImage;
	//flip(norm3, flipImage, 1);
	
	char frameNo[200];
	sprintf_s(frameNo, "%s\\%03i.tif", fileNameImage, i);
	ImgVector1.convertTo(ImgVector1, CV_8UC1);
	//cout << ImgVector.channels() << endl;
	imwrite(frameNo, ImgVector1);

}

Mat1b getMean1(const vector<Mat1b>& images)
{
	if (images.empty()) return Mat1b();
	// Create a 0 initialized image to use as accumulator
	Mat m(images[0].rows, images[0].cols, CV_32FC1);
	m.setTo(Scalar(0, 0, 0, 0));

	Mat temp;
	for (int i = 0; i < images.size(); ++i)
	{
		// Convert the input images to CV_64FC3 ...
		images[i].convertTo(temp, CV_32FC1);

		// accumulate
		m += temp;

	}
	// Convert back to CV_8UC3 type, applying the division to get the actual mean
	m.convertTo(m, CV_8U, 1. / images.size());
	cout << m << endl;
	return m;
}