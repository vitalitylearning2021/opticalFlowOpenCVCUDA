/***********************/
/* LUCAS-KANADE SPARSE */
/***********************/

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

/*********************************/
/* DEVICE TO HOST COPY FUNCTIONS */
/*********************************/
static void device2Host(const GpuMat &d_A, vector<Point2f> &h_A) {
	h_A.resize(d_A.cols);
	Mat mat(1, d_A.cols, CV_32FC2, (void *)&h_A[0]);
	d_A.download(mat); }

static void device2Host(const GpuMat &d_A, vector<uchar> &h_A) {
	h_A.resize(d_A.cols);
	Mat mat(1, d_A.cols, CV_8UC1, (void *)&h_A[0]);
	d_A.download(mat); }

/******************************/
/* DRAW OPTICAL FLOW FUNCTION */
/******************************/
static void drawFlow(Mat &frame, const vector<Point2f> &previousPoints, const vector<Point2f> &nextPoints, const vector<uchar> &status, Scalar line_color = Scalar(0, 0, 255)) {

	// --- Loop over all the points
	for (size_t i = 0; i < previousPoints.size(); ++i) {
		
		// --- Check if point status is ok
		if (status[i]) {
			
			// --- Set line thickness
			int line_thickness = 1;

			// --- Set previous and next points
			Point p = previousPoints[i];
			Point q = nextPoints[i];

			// --- Find angle and length of arrow
			double angle = atan2((double)p.y - q.y, (double)p.x - q.x);
			double hypotenuse = sqrt((double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x));

			if (hypotenuse < 1.0) continue;

			// --- If the length of the arrow is less than 1, then lengthen the arrow by a factor of three.
			q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
			q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

			// --- Define arrow line
			line(frame, p, q, line_color, line_thickness);

			// --- Draw the tips of the arrow. Some scaling is operated so that the tips look proportional to the 
			//     main line of the arrow.
			p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
			p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
			line(frame, p, q, line_color, line_thickness);

			p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
			p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
			line(frame, p, q, line_color, line_thickness);
		}
	}
}

/********/
/* MAIN */
/********/
int main() {

	// --- Images file names
	// --- https://ccv.wordpress.fos.auckland.ac.nz/data/stereo-pairs/
	string filename0 = "./rect_0384_c1.tif";
	string filename1 = "./rect_0385_c1.tif";

	Mat im0 = imread(filename0);
	Mat im1 = imread(filename1);

	cout << "Image size: " << im0.cols << " x " << im0.rows << endl;

	Mat im0Gray;
	//Mat im1Gray;

	// --- Converts images im0 and im1 to gray scale
	cv::cvtColor(im0, im0Gray, COLOR_BGR2GRAY);
	//cv::cvtColor(im1, im1Gray, COLOR_BGR2GRAY);

	/***********************************/
	/* SPOTTING GOOD FEATURES TO TRACK */
	/***********************************/
	GpuMat d_im0Gray(im0Gray);
	GpuMat d_previousPoints;

	// --- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_meaning/py_features_meaning.html
	// --- Ptr<CornersDetector> cv::cuda::createGoodFeaturesToTrackDetector(int srcType, int maxCorners = 1000,
	//		   					double qualityLevel = 0.01, double minDistance = 0.0, int blockSize = 3,
	//							bool useHarrisDetector = false, double harrisK = 0.04)	
	// --- srcType				Input source type. Only CV_8UC1 and CV_32FC1 are supported for now.
	// --- maxCorners			Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.
	// --- qualityLevel			Parameter characterizing the minimal accepted quality of image corners.The parameter 
	//							value is multiplied by the best corner quality measure, which is the minimal 
	//							eigenvalue (see cornerMinEigenVal) or the Harris function response (see cornerHarris).The corners with the quality measure less than the product are rejected.For example, if the best corner has the quality measure = 1500, and the qualityLevel = 0.01, then all the corners with the quality measure less than 15 are rejected.
	// --- minDistance			Minimum possible Euclidean distance between the returned corners.
	// --- blockSize			Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.See cornerEigenValsAndVecs .
	// --- useHarrisDetector	Parameter indicating whether to use a Harris detector (see cornerHarris) or 
	//							cornerMinEigenVal.
	// --- harrisK				Free parameter of the Harris detector.

	int		maxCorners = 4000;
	double	qualityLevel = 0.01;
	double	minDistance = 0;
	int		blockSize = 5;

	Ptr<cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(d_im0Gray.type(), maxCorners, qualityLevel,
		minDistance, blockSize);
	detector->detect(d_im0Gray, d_previousPoints);

	/**************************************/
	/* LUCAS-KANADE'S SPARSE OPTICAL FLOW */
	/**************************************/
	GpuMat d_frame0(im0);
	GpuMat d_frame1(im1);
	//GpuMat d_frame1Gray(im1Gray);
	GpuMat d_nextPoints;
	GpuMat d_status;
	//GpuMat d_flow(im0.size(), CV_32FC2);

	int		winSize = 21;
	int		maxLevel = 3;
	int		iters = 30;
	Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(winSize, winSize), maxLevel, iters);

	// --- SparseOpticalFlow::calc(InputArray prevImg, InputArray nextImg, InputArray prevPts, InputOutputArray	
	//		nextPts, OutputArray status, OutputArray err = cv::noArray())
	// --- prevImg				First input image.
	// --- nextImg				Second input image of the same size and the same type as prevImg.
	// --- prevPts				Vector of 2D points for which the flow needs to be found.
	// --- nextPts				Output vector of 2D points containing the calculated new positions of input features 
	//							in the second image.
	// --- status				Output status vector.Each element of the vector is set to 1 if the flow for the 
	//							corresponding features has been found. Otherwise, it is set to 0.
	// --- err					Optional output vector that contains error response for each point (inverse 
	//							confidence).
	d_pyrLK_sparse->calc(d_frame0, d_frame1, d_previousPoints, d_nextPoints, d_status);

	// --- Copies from device to host
	vector<Point2f> h_previousPoints(d_previousPoints.cols);
	device2Host(d_previousPoints, h_previousPoints);

	vector<Point2f> h_nextPoints(d_nextPoints.cols);
	device2Host(d_nextPoints, h_nextPoints);

	vector<uchar> status(d_status.cols);
	device2Host(d_status, status);

	// --- Draw optical flow
	namedWindow("PyrLK [Sparse]", WINDOW_NORMAL);
	drawFlow(im0, h_previousPoints, h_nextPoints, status, Scalar(255, 0, 0));
	imshow("PyrLK [Sparse]", im0);

	waitKey(0);

	return 0;
}
