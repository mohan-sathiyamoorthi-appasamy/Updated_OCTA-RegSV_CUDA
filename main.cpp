
#include "Corr_Axial.h"

int main(int argc, char** argv)
{

	//*********************************************************************************//
	//Read BM Image Stack Size 256x256x384
	char *filename = new char[100];
	filename = argv[1];
	printf("Binary Input: %s\n", filename);
	char *RegDirName = argv[2];
	printf("RegDirName: %s\n", RegDirName);
	char *SVDirName = argv[3];
	printf("SVDirName: %s\n", SVDirName);

	FILE* file = fopen(filename, "rb");
	if (file == NULL)
	{
		printf("Unable to Open File\n");
		exit(1);
	}
	fseek(file, 0, SEEK_END);

	int host_fileLen;
	host_fileLen = ftell(file) / (int)sizeof(unsigned short);
	printf("Host Memory File Length:%d\n", host_fileLen);
	rewind(file);

	//Memory alloction for Host array
	unsigned short* h_Image_Stack;
	h_Image_Stack = (unsigned short*)malloc(host_fileLen * sizeof(unsigned short));
	fread(h_Image_Stack, 1, host_fileLen * sizeof(unsigned short), file);
	fclose(file);


	// Type Conversion - ushort to float
	// Allocate Device Memory for ushort input Data
	unsigned short *d_ushort_input = NULL;
	cudaMalloc((void**)&d_ushort_input, host_fileLen * sizeof(unsigned short));
	cudaMemset(d_ushort_input, 0, (host_fileLen)* sizeof(unsigned short));
	cudaMemcpy(d_ushort_input, h_Image_Stack, host_fileLen * sizeof(unsigned short), cudaMemcpyHostToDevice);

	/*for (int i = 0; i <host_fileLen; i++)
	{
	printf("%2u\n", h_Image_Stack[i]);
	}*/

	//Memory Free
	free(h_Image_Stack);

	//**********************************************************************************//
	// Volume Details
	int numFrames = 2;
	int width = 500;
	int height = 975;
	int inputFrameCount;
	int imageSize = width * height;
	//************************************************************************************//

	//char* RegFilePath = argv[2];
	//char* SVFilePath = argv[3];


	// UNSIGNED SHORT  to FLOAT CONVERSION
	//	Allocate Device Memory 
	float *d_float_input = NULL;
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void**)&d_float_input, numFrames * imageSize * sizeof(float));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device Memory(error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Crop the Image
	float *mask = NULL;
	cudaMalloc((void**)&mask, imageSize * sizeof(float));

	// Size Modification to 2^n
	int total_num_elements = imageSize;
	int value = round(log(total_num_elements) / log(2));
	int num_elements = pow(2, value);

	float *mask_In = NULL;
	cudaMalloc((void**)&mask_In, num_elements * sizeof(float));

	float *field = NULL;
	cudaMalloc((void**)&field, imageSize * sizeof(float));   

	float *field_In = NULL;
	cudaMalloc((void**)&field_In, num_elements * sizeof(float));

	float *corr_result_stack = NULL;
	corr_result_stack = (float*)malloc(41 * sizeof(float));

	float *motionA = NULL;
	motionA = (float*)malloc(numFrames*sizeof(float));
	memset(motionA, 0, numFrames*sizeof(float));


	//*****************************************************************************************************************************************//
	float romA = 20.0;
	int frames = 1;
	int frameIdx = 0;
	int offset = 20;
	int range = 935;
	int Total_Offset = 40;
	//*****************************************************************************************************************************************//


	float corr_result_value = 0;
	//clock_t cpu_start = clock();
	//cout << "cpu Elapsed start time is" << cpu_start << endl;

	cudaStream_t stream[384];
	cudaStream_t stream1[40];


	for (inputFrameCount = 0; inputFrameCount < numFrames; inputFrameCount += 1)
	{
		cudaStreamCreate(&stream[inputFrameCount]);
		//uShort To Float Conversion
		uShortToFloatConversion(&d_ushort_input[(inputFrameCount)*(imageSize)], &d_float_input[(inputFrameCount)*(imageSize)], imageSize, stream[inputFrameCount]);
		cudaDeviceSynchronize();
	}


	for (int inputFrameCount1 = 1; inputFrameCount1 < numFrames; inputFrameCount1 += 1)
	{
		cudaStreamCreate(&stream[inputFrameCount - 1]);

		// Image 1 - Mask Image
		postFFTCrop(&d_float_input[(inputFrameCount1 - 1)*(imageSize)], mask, 1, frameIdx, offset, range, stream[inputFrameCount - 1]);
		cudaDeviceSynchronize();

		// Size Modification for Mean & Block Sum Calculation
		SizeModification(mask_In, mask, num_elements, imageSize, stream[inputFrameCount - 1]);
		cudaDeviceSynchronize();




		// Image 2 - Field Image
		for (int offset = 0; offset <= Total_Offset; offset++)
		{
			cudaStreamCreate(&stream1[offset]);

			postFFTCrop(&d_float_input[(inputFrameCount1)*(imageSize)], field, 1, frameIdx, offset, range, stream1[offset]);
			cudaDeviceSynchronize();

			SizeModification(field_In, field, num_elements, imageSize, stream1[offset]);
			cudaDeviceSynchronize();


			corr_result_value = corrcoeff_Calculation(mask_In, field_In, (935 * 500), num_elements, stream1[offset]);
			 
			corr_result_stack[offset] = abs(corr_result_value);
		}

		//Find Index of Maximum Value
		float max_Location = 0;
		max_Location = IndexMaxElement(corr_result_stack, Total_Offset);


		//printf("%d %2f\n", inputFrameCount1, max_Location);
		//Axial Motion Vector
		motionA[inputFrameCount1] = max_Location - romA - 1;
		motionA[inputFrameCount1] = motionA[inputFrameCount1] + motionA[inputFrameCount1 - 1];
		//cout << motionA[inputFrameCount1] << endl;
	}


	for (int i = 0; i < 40; i++)
		cudaStreamSynchronize(stream1[i]);


	//clock_t cpu_stop = clock();
	//clock_t time_spent = 1000 * (cpu_stop - cpu_start) / CLOCKS_PER_SEC;
	//cout << "time_spent" << time_spent << endl;

	//Memory Free
	cudaFree(d_float_input);
	cudaFree(mask);
	cudaFree(mask_In);
	cudaFree(field);
	cudaFree(field_In);

	//*********************************************************************************************************************************************//
	//// Polynomial Fit
	double t = (double)getTickCount();
	Mat img(1, numFrames, CV_32FC1, motionA);
	std::vector<cv::Point> points;
	//std::cout << "A = " << img.at<float>(383) << std::endl;

	for (int index = 0; index < numFrames; index++)
	{
		points.push_back(cv::Point(index, img.at<float>(index)));
	}

	cv::Mat A;
	// Calling Polynomial Function
	polynomial_curve_fit(points, 2, A);
	//std::cout << "A = " << A << std::endl;

	std::vector<cv::Point> points_fitted;
	double *y = NULL;
	y = (double*)malloc(numFrames * sizeof(double));

	for (int x = 0; x < numFrames; x++)
	{
		y[x] = A.at<double>(0, 0) + A.at<double>(1, 0) * x + A.at<double>(2, 0)*std::pow(x, 2);
		//cout << y[x] << endl;
		//points_fitted.push_back(cv::Point(x, y));
	}
	t = ((double)getTickCount() - t) / getTickFrequency();
	//std::cout << "Times passed in seconds: " << t << std::endl;
	//*********************************************************************************************************************************************//
	// Compute Motion Correction Parameters and do Motion Correction
	float *disp_Ind = NULL;
	disp_Ind = (float*)malloc(numFrames * sizeof(float));

	for (int i = 0; i < numFrames; i++)
	{
		disp_Ind[i] = (float)(motionA[i] - y[i]);
	}


	// Find Maximum Value of subtracted value of  Motion Value 
	float topZero = 0;
	topZero = FindMaxValue(disp_Ind, numFrames);


	// Find Minimum and Absolute value
	float botZero = 0;
	botZero = FindAbsMinValue(disp_Ind, numFrames);

	// Subtraction top_zero and disp_Ind
	float *top = NULL;
	top = (float*)malloc(numFrames * sizeof(float));

	for (int i = 0; i < numFrames; i++)
	{
		top[i] = round(topZero - disp_Ind[i]);

	}

	//******************************************************************************************************************************************//

	// Crop Image
	// Find Maximum Value from top data to assign Hightest Height of data
	int height_Offset;
	height_Offset = FindMaxValue(top, numFrames);
	//cout << height_Offset << endl;

	int new_Height = 0;
	new_Height = height + height_Offset;

	int crop_off = topZero + botZero;


	float *volume_mcorr = NULL;
	err = cudaMalloc((void**)&volume_mcorr, (new_Height*width)*sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemset(volume_mcorr, 0, (new_Height*width)*sizeof(float));




	float *d_float_input_copy = NULL;
	err = cudaMalloc((void**)&d_float_input_copy, imageSize * sizeof(float));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device Memory1(error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}



	float *volume_mcorr_crop = NULL;
	err = cudaMalloc((void**)&volume_mcorr_crop, (imageSize)*sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device Memory2(error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemset(volume_mcorr_crop, 0, (imageSize)*sizeof(float));


	int crop_Level = 0;
	int new_Image_Height = 867;

	float *volume_mcorr_crop2 = NULL;
	cudaMalloc((void**)&volume_mcorr_crop2, (new_Image_Height*width*numFrames)*sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device Memory3(error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaMemset(volume_mcorr_crop2, 0, (new_Image_Height*width*numFrames)*sizeof(float));


	// Float Conversion Kernel
	for (inputFrameCount = 0; inputFrameCount < numFrames; inputFrameCount += 1)
	{
		floatConversionProcess(&d_ushort_input[(inputFrameCount)*(imageSize)], d_float_input_copy, imageSize);

		Image_Assign_NewSize(d_float_input_copy, volume_mcorr, new_Height*width, top[inputFrameCount]);

		crop_level1(volume_mcorr, volume_mcorr_crop, new_Height*width, crop_off);

		crop_level2(volume_mcorr_crop, &volume_mcorr_crop2[(inputFrameCount)*(new_Image_Height*width)], imageSize, crop_Level);
	}

	for (int i = 0; i < 384; i++)
		cudaStreamSynchronize(stream[i]);

	//**************************************************************************************************************************************************************//
	int subPixelFactor = 2;

	//2.DFT Registration Process//

	for (inputFrameCount = 0; inputFrameCount < 384; inputFrameCount += 3)
	{

		dftregistration(&volume_mcorr_crop2[(inputFrameCount)*(new_Image_Height*width)], subPixelFactor, new_Image_Height, width, 3, 0);//numF = 3; frameNum=0

	}


	for (inputFrameCount = 1; inputFrameCount <= 384; inputFrameCount += 1)
	{

		cudaMemset(&volume_mcorr_crop2[(inputFrameCount)*(new_Image_Height*width) - new_Image_Height], 0, 867 * sizeof(float));

	}



	//Save Batch Average Image
	int numFrames3 = 384;

	float *test3 = NULL;
	test3 = (float*)malloc(new_Image_Height*width*numFrames3* sizeof(float));
	if (test3 == NULL)
	{
		printf("Memory allocation failed");

	}
	memset(test3, 0, new_Image_Height*width*numFrames3* sizeof(float));
	cudaMemcpy(test3, volume_mcorr_crop2, new_Image_Height*width*numFrames3*sizeof(float), cudaMemcpyDeviceToHost);

	//const char *fileName = "C:\\Users\\Mohan\\Documents\\Visual Studio 2013\\Projects\\OCT_A_Complete_Project\\OCT_A_Complete_Project\\TEST.txt";
	int ctr = 0;
	
	//Average
	for (int j = 0; j < 384; j += 3)
	{

		vector<Mat> images;
		for (int i = 0; i < 3; i++)
		{
			Mat Image(width,new_Image_Height, CV_32FC1);

			memcpy(Image.data, &test3[(i + j)*(new_Image_Height*width)], new_Image_Height*width*sizeof(float));
			//writeMatToFile(Image, fileName);
			images.push_back(Image);

		}
		 
		Mat add_meanimg = getMean(images,RegDirName,new_Image_Height*width,ctr);

		
		ctr = ctr + 1;
		
	}
		/*for (int i = 0; i <384; i++)
		{
		SaveTifImage(&test3[(i)*(new_Image_Height*width)], RegDirName, new_Image_Height*width, i);	
		}*/

		//free(singleImage);
		free(test3);

		//****************************************************************************************************************************************************//

		float *specVar_result;
		err = cudaMalloc((void**)&specVar_result, (width * new_Image_Height * 128) * sizeof(float));
		cudaMemset(specVar_result, 0, width * new_Image_Height * 128 * sizeof(float));
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device Memory3(error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}



		//3.Invoke Speckle Variance
		//int inputFrameCount = 3;
		int i = 0;
		for (inputFrameCount = 0; inputFrameCount < 383; inputFrameCount += 3)
		{
			speckleVar(&volume_mcorr_crop2[(inputFrameCount)*(width*new_Image_Height)], &specVar_result[(inputFrameCount / 3)*(width*new_Image_Height)], width, new_Image_Height, 3, 0);
			i = i + 1;
		}




		//Save Speckle Variance Image
		//Test	
		int numFrames2 = 128;

		float *test2 = NULL;
		test2 = (float*)malloc(new_Image_Height*width*numFrames2* sizeof(float));
		if (test2 == NULL)
		{
			printf("Memory allocation failed");
		
		}
		 
		cudaMemcpy(test2, specVar_result, new_Image_Height*width*numFrames2* sizeof(float), cudaMemcpyDeviceToHost);
		




		//const char *fileName = "C:\\Users\\Mohan\\Documents\\Visual Studio 2013\\Projects\\OCT_A_Complete_Project\\OCT_A_Complete_Project\\TEST.txt";
	 
		//for (int i = 0; i <128; i++)
		//{
			//SaveTifImage(&test2[(i)*(new_Image_Height*width)],SVDirName, new_Image_Height*width, i);
		//Mat1b Image(new_Image_Height, width, CV_32FC1);
		//	memcpy(Image.data, &test2[(i)*(new_Image_Height*width)], new_Image_Height*width*sizeof(float));
			//writeMatToFile(Image, fileName);
		//}


		free(y);
		free(motionA);
		free(corr_result_stack);
		free(disp_Ind);

		cudaFree(d_float_input_copy);
		cudaFree(volume_mcorr);
		cudaFree(volume_mcorr_crop);
		cudaFree(specVar_result);
		free(test2);
		cudaFree(volume_mcorr_crop2);
		return 0;
	}
	 
	void writeMatToFile(cv::Mat m, const char* filename)
	{
		ofstream fout(filename);

		if (!fout)
		{
			cout << "File Not Opened" << endl;  return;
		}

		for (int i = 0; i<m.rows; i++)
		{
			for (int j = 0; j<m.cols; j++)
			{
				fout << m.at<float>(i, j) << "\t";
			}
			fout << endl;
		}

		fout.close();
	}



