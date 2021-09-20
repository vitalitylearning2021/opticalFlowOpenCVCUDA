#include <iostream>
#include <fstream>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

// --- Relative lengths of color transitions, chosen according to perceptual similarity (e.g. one can distinguish 
//     more shades between red and yellow than between yellow and green)
const int RY		= 15;
const int YG		= 6;
const int GC		= 4;
const int CB		= 11;
const int BM		= 13;
const int MR		= 6;
const int NSHADES	= RY + YG + GC + CB + BM + MR;

// --- Color wheel
static Vec3i colorWheel[NSHADES];

/********************************/
/* COMPUTE COLOR WHEEL FUNCTION */
/********************************/
void computeColorWheel() {

	int k = 0;
	for (int i = 0; i < RY; ++i, ++k) colorWheel[k] = Vec3i(255,				255 * i / RY,		0);
	for (int i = 0; i < YG; ++i, ++k) colorWheel[k] = Vec3i(255 - 255 * i / YG, 255,				0);
	for (int i = 0; i < GC; ++i, ++k) colorWheel[k] = Vec3i(0,					255,				255 * i / GC);
	for (int i = 0; i < CB; ++i, ++k) colorWheel[k] = Vec3i(0,					255 - 255 * i / CB, 255);
	for (int i = 0; i < BM; ++i, ++k) colorWheel[k] = Vec3i(255 * i / BM,		0,					255);
	for (int i = 0; i < MR; ++i, ++k) colorWheel[k] = Vec3i(255,				0,					255 - 255 * i / MR);
}

/********************/
/* VALID FLOW CHECK */
/********************/
inline bool isFlowValid(Point2f u) { return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9; }

/*****************************/
/* RETURN HUE COLOR FUNCTION */
/*****************************/
static Vec3b returnCOLOR(float ux, float uy) {

	// --- Displacement: distance from origin
	const float rad				= sqrt(ux * ux + uy * uy);
	// --- Displacement: angular position normalized by pi (angular position comprised in (-1, 1)
	const float angleNormalized	= atan2(-uy, -ux) / (float)CV_PI;

	// --- Divides the full circle into NSHADES slices and computes the position of the displacement vector
	//     within such slices
	const float shadingPosition	= (angleNormalized + 1.0f) / 2.0f * (NSHADES - 1);
	// --- Beginning slice index
	const int	shade0	= static_cast<int>(shadingPosition);
	// --- Ending slice index
	const int	shade1	= (shade0 + 1) % NSHADES;
	// --- Offset wrt beginning slice index
	const float f	= shadingPosition - shade0;

	Vec3b displacementColor;

	for (int RGBcol = 0; RGBcol < 3; RGBcol++)
	{
		// --- Computes normalized RGB color corresponding to initial shade
		const float col0 = colorWheel[shade0][RGBcol] / 255.0f;
		// --- Computes normalized RGB color corresponding to final shade
		const float col1 = colorWheel[shade1][RGBcol] / 255.0f;

		// --- Computes hue of HSV according to angular position of displacement
		float col = (1 - f) * col0 + f * col1;

		if (rad <= 1)
			// --- Change saturation with radius. For rad = 0, col = 1; for rad = 1, col = col.
			col = 1 - rad * (1 - col); 
		else
			// --- Radius out of range
			col *= .75; 

		displacementColor[RGBcol] = static_cast<uchar>(255.0 * col);
	}

	return displacementColor;
}

/**************************/
/* COLORING FLOW FUNCTION */
/**************************/
static void colorOpticalFlow(const Mat_<float> &h_dx, const Mat_<float> &h_dy, Mat &h_coloredFlow, float maxmotion)
{
	// --- Creates a colored flow of the same size of the flow and sets it to all zeros
	h_coloredFlow.create(h_dx.size(), CV_8UC3);
	h_coloredFlow.setTo(Scalar::all(0));

	for (int y = 0; y < h_dx.rows; ++y) {
		for (int x = 0; x < h_dx.cols; ++x) {
			
			Point2f u(h_dx(y, x), h_dy(y, x));

			// --- In the flow is valid, returns the color associated to the displacement
			if (isFlowValid(u)) h_coloredFlow.at<Vec3b>(y, x) = returnCOLOR(u.x / maxmotion, u.y / maxmotion); }}}

/**********************************/
/* COMPUTE FLOW AND SHOW FUNCTION */
/**********************************/
static void computeFlowAndShow(const char *name, const GpuMat &d_opticalFlow)
{
	// --- Split the x and y components of the displacement of a two channel matrix into an array of matrices
	GpuMat planes[2];
	cuda::split(d_opticalFlow, planes);

	// --- Copy the x and y flow components to CPU
	Mat opticalFlowx(planes[0]);
	Mat opticalFlowy(planes[1]);

	Mat out;
	colorOpticalFlow(opticalFlowx, opticalFlowy, out, 10);

	imshow(name, out);
}

/**********************/
/* FILECHECK FUNCTION */
/**********************/
int fileCheck(Mat &im0, Mat &im1, string &filename1, string &filename2) {
	
	if (im0.empty())
	{
		cerr << "Image file [" << filename1 << "] can't be opened. Please, check." << endl;
		return -1;
	}
	
	if (im1.empty())
	{
		cerr << "Image file [" << filename2 << "] can't be opened. Please, check." << endl;
		return -1;
	}

	if (im1.size() != im0.size())
	{
		cerr << "Images are not of equal size. Please, check." << endl;
		return -1;
	}
}

/********/
/* MAIN */
/********/
int main() {
	
	// --- Images file names
	// --- https://ccv.wordpress.fos.auckland.ac.nz/data/stereo-pairs/
	string filename1 = "./rect_0384_c1.tif";
	string filename2 = "./rect_0385_c1.tif";

	// --- Loading images into OpenCV matrices
	Mat im0 = imread(filename1, IMREAD_GRAYSCALE);
	Mat im1 = imread(filename2, IMREAD_GRAYSCALE);

	// --- Checking wether files can be opened or images have the same size
	const int fileCheckInt = fileCheck(im0, im1, filename1, filename2);
	if (fileCheckInt == -1) return -1;
		 
	// --- Moving images from CPU to GPU
	GpuMat d_im0(im0);
	GpuMat d_im1(im1);

	GpuMat d_opticalFlow(im0.size(), CV_32FC2);

	// --- Compute color wheel
	computeColorWheel();

	/*************/
	/* FARNEBACK */
	/*************/
	// ---  cuda::FarnebackOpticalFlow::create(int numLevels=5, double pyrScale=0.5, bool fastPyramids=false, int winSize=13, int numIters=10, int polyN=5, double polySigma=1.1, int flags=0)
	//		numLevels			= number of pyramid layers including the initial image; levels=1 means that no extra 
	//							  layers are created and only the original images are used.
	//		pyr_scale			= parameter, specifying the image scale (<1) to build pyramids for each image; 
	//							  pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller 
	//							  than the previous one.
	//		fastPyramids		= use fast pyramids approach
	//		winsize				= averaging window size; larger values increase the algorithm robustness to image 
	//							  noise and give more chances for fast motion detection, but yield more blurred 
	//							  motion field.
	//      numIters			= number of iterations the algorithm does at each pyramid level.
	//		polyN 				= size of the pixel neighborhood used to find polynomial expansion in each pixel; 
	//							  larger values mean that the image will be approximated with smoother surfaces, 
	//						      yielding more robust algorithm and more blurred motion field, typically poly_n=5 or 7.
	//      polySigma			= standard deviation of the Gaussian that is used to smooth derivatives used as a 
	//							  basis for the polynomial expansion; for polyN=5, you can set polySigma=1.1, 
	//							  for polyN=7, a good value would be polySigma=1.5.
	//		flags				= operation flags that can be a combination of the following:
	//							  OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
	//							  OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize x winsize filter instead of 
	//							  a box filter of the same size for optical flow estimation; usually, this option 
	//							  gives z more accurate flow than with a box filter, at the cost of lower speed; normally, winsize for a Gaussian window should be set to a larger value to achieve the same level of robustness.
	Ptr<cuda::FarnebackOpticalFlow>		farn = cuda::FarnebackOpticalFlow::create(6, 0.4, false, 13, 15, 5, 1.1, 256);
	{
		farn->calc(d_im0, d_im1, d_opticalFlow);

		computeFlowAndShow("Farnebäck", d_opticalFlow);
	}

	/*********************/
	/* BROX OPTICAL FLOW */
	/*********************/
	// ---	cuda::BroxOpticalFlow::create(double alpha=0.197, double gamma=50.0, double scale_factor=0.8, int inner_iterations=5, int outer_iterations=150, int solver_iterations=10)
	//		alpha				= flow smoothness functional weight
	//		gamma				= gradient constancy importance functional weight
	//		scale_factor		= pyramid scale factor belonging to (0,1)
	//		inner_iterations	= number of lagged non-linearity iterations (inner loop)
	//		outer_iterations	= number of pyramid levels
	//		solver_iterations	= number of linear system solver iterations
	Ptr<cuda::BroxOpticalFlow>			brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
	{
		GpuMat d_im0Scaled;
		GpuMat d_im1Scaled;

		d_im0.convertTo(d_im0Scaled, CV_32F, 1.0 / 255.0);
		d_im1.convertTo(d_im1Scaled, CV_32F, 1.0 / 255.0);

		brox->calc(d_im0Scaled, d_im1Scaled, d_opticalFlow);

		computeFlowAndShow("Brox et al.", d_opticalFlow);
	}

	/**************/
	/* DUAL TV-L1 */
	/**************/
	// ---  cuda::OpticalFlowDual_TVL1::create(double tau=0.25, double lambda=0.15, double theta=0.3, int nscales=5, int warps=5, double epsilon=0.01, int iterations=300, double scaleStep=0.8, double gamma=0.0, bool useInitialFlow=false)
	//		tau					= time step of the numerical scheme
	//		lambda				= weight parameter for the data term, attachment parameter. This is the most relevant 
	//                            parameter, which determines the smoothness of the output. The smaller this 
	//							  parameter is, the smoother the solutions we obtain. It depends on the range of 
	//							  motions of the images, so its value should be adapted to each image sequence.
	//		theta				= weight parameter for (u - v)^2, tightness parameter. It serves as a link between 
	//							  the attachment and the regularization terms. In theory, it should have a small 
	//							  value in order to maintain both parts in correspondence. The method is stable for 
	//							  a large range of values of this parameter.
	//		nscales				= number of scales used to create the pyramid of images.
	//		warps				= number of warpings per scale. Represents the number of times that I1(x+u0) and 
	//						      grad( I1(x+u0) ) are computed per scale. This is a parameter that assures the 
	//							  stability of the method. It also affects the running time, so it is a compromise 
	//							  between speed and accuracy.
	//		epsilon				= stopping criterion threshold used in the numerical scheme, which is a trade-off 
	//							  between precision and running time. A small value will yield more accurate 
	//							  solutions at the expense of a slower convergence.	
	//		iterations			= stopping criterion iterations number used in the numerical scheme.
	//		scaleStep			= Step between scales (<1).
	//		gamma				= *q +.
	//		useInitialFlow		= Use initial flow.
	Ptr<cuda::OpticalFlowDual_TVL1>		tvl1 = cuda::OpticalFlowDual_TVL1::create();
	// Ptr<cuda::OpticalFlowDual_TVL1>		tvl1 = cuda::OpticalFlowDual_TVL1::create(0.25, 0.15, 0.3, 5, 5, 0.01, 300, 0.8, 10.0, false);
	{
		tvl1->calc(d_im0, d_im1, d_opticalFlow);

		computeFlowAndShow("TVL1", d_opticalFlow);
	}

	/****************/
	/* LUCAS-KANADE */
	/****************/
	// ---  cv::cuda::DensePyrLKOpticalFlow::create(Size winSize = Size(13, 13), int maxLevel = 3, int iters = 30, bool useInitialFlow = false)
	//		winSize				= size of the search window at each pyramid level
	//		maxLevel			= 0-based maximal pyramid level number; if set to 0, pyramids are not used (single 
	//							  level), if set to 1, two levels are used, and so on; if pyramids are passed to 
	//							  input then algorithm will use as many levels as pyramids have but no more than 
	//							  maxLevel.
	//		iters				= number of iterations according to http://www.ieee-hpec.org/2014/CD/index_htm_files/FinalPapers/98.pdf
	//		useInitialFlow		= exploit User's provided initial flow as starting guess
	Ptr<cuda::DensePyrLKOpticalFlow>	lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));
	{
		lk->calc(d_im0, d_im1, d_opticalFlow);

		computeFlowAndShow("Lucas-Kanade", d_opticalFlow);
	}

	imshow("Frame 0", im0);
	imshow("Frame 1", im1);
	waitKey();

	return 0;
}
