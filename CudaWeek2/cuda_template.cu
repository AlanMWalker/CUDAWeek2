/*/*
* 55-604127 Tools, Libraries and Frameworks
Template for problem 2
The following contains the sequential code for the image manipulation problem
It is also a template for the CUDA version
You should use it as reference for performance analisys
*/

#define THREADS_PER_BLOCK 32

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "CImg.h"
#include "time.h"
using namespace cimg_library;

__global__ void grayscale_kernel(const unsigned char *input, unsigned char *output, int imagesize)
{
	const long index = threadIdx.x + blockIdx.x * blockDim.x;

	const long offsetG = imagesize;
	const long offsetB = imagesize * 2;

	const unsigned long roffset = index;
	const unsigned long goffset = index + offsetG;
	const unsigned long boffset = index + offsetB;

	const unsigned char r = input[roffset];
	const unsigned char g = input[goffset];
	const unsigned char b = input[boffset];

	output[index] = (char)((0.33f * r) + (0.33f*g) + (0.33f*b));
}

int main()
{
	//Creation of one instances of images of unsigned char pixels. 
	//The first image image is initialized by reading an image file from the disk. 
	//Here, lena.bmp must be in the same directory as the current program. Note that you may have installed other packages in order to read different image types (e.g. jpg)
	//This is a sample image of 512*512, try other of different size, i.e. smaller, bigger...
	CImg<unsigned char> rgbimage("lena512colour.bmp");
	// The second image is the output
	CImg<unsigned char> grayscale(rgbimage.width(), rgbimage.height(), 1, 1 /*one because grayscale has only one channel*/, 0);

	// CImg is a template, images are stored as vectors, you need to retrieve the pointer to the first item
	const unsigned char *rgbimage_pointer = rgbimage.data();
	unsigned char *grayscale_pointer = grayscale.data();

	//calculates the offsets. The library allocates first the reds, then green and blues.
	//in practice we have three contigous vectors of width*height
	const long offsetr = rgbimage.offset(0, 0, 0, 0); // this is zero because the first vector is red
	const long offsetg = rgbimage.offset(0, 0, 0, 1); // this will be width*height
	const long offsetb = rgbimage.offset(0, 0, 0, 2); // this will be 2*width*height
	// offset for the grayscale
	const long offsety = grayscale.offset(0, 0, 0, 0); // this is zero

	// prints the offset for double checking
	printf("%i ; %i ; %i \n", offsetr, offsetg, offsetb);
	printf("%i", offsety);

	/* this organisation is not efficient, would be more efficient to have the RGB values for each pixel one ofter the other,
	e.g R0G0B0,R1G1B1,R2G2B2,etc... can you do it? this will increase peformance see the memory coalescing slides*/

	// THE FOLLOWING LOOP WILL RUN FASTER IN PARALLEL? 
	clock_t start_cpu = clock();
	for (int w = 0; w < rgbimage.width(); w++)
	{
		for (int h = 0; h < rgbimage.width(); h++)
		{
			unsigned int grayOffset = w*rgbimage.height() + h;
			unsigned int rOffset = grayOffset + offsetr;
			unsigned int gOffset = grayOffset + offsetg;
			unsigned int bOffset = grayOffset + offsetb;
			//retrieves the pixel values
			unsigned char r = rgbimage_pointer[rOffset]; // red value for pixel
			unsigned char g = rgbimage_pointer[gOffset]; // green value for pixel
			unsigned char b = rgbimage_pointer[bOffset]; // blue value for pixel
			// performs the rescaling and store it, multiply by floating point constants
			grayscale_pointer[grayOffset] = (char)(0.33f*r + 0.33f*g + 0.33f*b);

			// Other grayscale conversions, try them too
			//(0.299*R + 0.587*G + 0.114*B);
			// 0.21f*R + 0.71f*G + 0.07f*B;
		}
	}
	//for timing. It is required to run this at least 10 times and use the average, the median and/or the worst case in the analysis
	clock_t end_cpu = clock();
	double cpu_time = (((double)(end_cpu - start_cpu) / CLOCKS_PER_SEC)) * 1000.0;

	printf("\nTime elapsed by CPU: %f ms\n\n", cpu_time);

	// the following displays both images
	CImgDisplay main_disp(rgbimage, "Original"), draw_dispGr(grayscale, "Grayscale");

	/*GPU version below*/

	/*DECLARE 2 POINTERS FOR ALLOCATING GPU MEMORY: RGBIMAGE AND GRAYSCALE*/
	cudaEvent_t start;
	cudaEvent_t stop;
	float elapsedTime, elapsedTime1, elapsedTime2;

	CImg<unsigned char> gImgRGB("lena512colour.bmp");
	CImg<unsigned char> gImgGrey(gImgRGB._width, gImgRGB._height, 1, 1, 0);
	const UINT ImageSize = gImgRGB._width * gImgRGB._height;

	//Pointers to memory on GPU
	unsigned char* d_pImgRGBData = nullptr;
	unsigned char* d_pImgGreyData = nullptr;

	cudaMalloc((void **)&d_pImgRGBData, 3 * ImageSize  * sizeof(char));
	cudaMalloc((void **)&d_pImgGreyData, ImageSize  * sizeof(char));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	/* copy inputs to device */
	/* fix the parameters needed to copy data to the device */
	cudaMemcpy(d_pImgRGBData, gImgRGB._data, 3 * ImageSize  * sizeof(char), cudaMemcpyHostToDevice);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime1, start, stop);

	cudaEventRecord(start, 0);

	//const int image_size = rgbimage.width() * rgbimage.height();
	/* launch the kernel on the GPU */
	/* insert the launch parameters to launch the kernel properly using blocks and threads */
	grayscale_kernel << < gImgRGB._width, gImgRGB._height >> >(d_pImgRGBData, d_pImgGreyData, ImageSize);

	auto cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	/* copy result back to host */
	cudaEventRecord(start, 0);
	/* fix the parameters needed to copy data back to the host */
	cudaMemcpy(gImgGrey._data, d_pImgGreyData, ImageSize * sizeof(char), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime2, start, stop);

	printf("Elapsed time on GPU: %f ms\n", elapsedTime2);
		
	/*do the performance analysis here*/
	/* for inspiration check the Vector addition exercise solution (example_cpu_gpu_timing(2).cu) */

	// the following displays the GPU version
	CImgDisplay draw_dispGPU(gImgGrey, "GrayscaleGPU");

	// wait until main window is closed
	while (!main_disp.is_closed())
	{
		main_disp.wait();
	}

	return 0;
}