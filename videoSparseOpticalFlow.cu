#include <  iostream>    
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\cudaobjdetect.hpp"
#include "opencv2\cudaimgproc.hpp"
#include "opencv2\cudawarping.hpp"
#include <  opencv2\bgsegm.hpp>  
#include <  opencv2\cudabgsegm.hpp>  
#include <  opencv2\cudaoptflow.hpp>  
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
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
void main() {
	
	// --- CPU images
	Mat im0, im1;

	// --- GPU images
	cuda::GpuMat d_im0Gray, d_im1Gray, d_frame0, d_frame1;

	// --- https://pixabay.com/videos/car-traffic-daytime-driving-on-road-16849/
	VideoCapture cap("Car - 16849.mp4");
	cap >> im0;
	if (im0.empty()) return;

	// --- Scaling factor
	//double scale = 800. / im0.cols;
	double scale = 1;

	// --- First image
	d_frame0.upload(im0);
	cuda::resize(d_frame0, d_im0Gray, Size(d_frame0.cols * scale, d_frame0.rows * scale));
	cuda::cvtColor(d_im0Gray, d_im0Gray, COLOR_BGR2GRAY);

	/***********************************/
	/* SPOTTING GOOD FEATURES TO TRACK */
	/***********************************/
	int		maxCorners		= 4000;
	double	qualityLevel	= 0.01;
	double	minDistance		= 0;
	int		blockSize		= 5;
	Ptr<cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(d_im0Gray.type(), maxCorners, qualityLevel, minDistance, blockSize);

	/**************************************/
	/* LUCAS-KANADE'S SPARSE OPTICAL FLOW */
	/**************************************/
	cuda::GpuMat d_previousPoints;
	cuda::GpuMat d_nextPoints;
	cuda::GpuMat d_status;

	int		winSize			= 21;
	int		maxLevel		= 3;
	int		iters			= 30;
	Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK = cuda::SparsePyrLKOpticalFlow::create(Size(winSize, winSize), maxLevel, iters);

	while (1) {

		// --- Get new image
		cap >> im1;
		if (im1.empty()) break;

		d_frame1.upload(im1);
		cuda::resize(d_frame1, d_im1Gray, Size(d_frame1.cols * scale, d_frame1.rows * scale));
		d_im1Gray.download(im1);
		// --- Converts image to gray scale
		cuda::cvtColor(d_im1Gray, d_im1Gray, COLOR_BGR2GRAY);

		// --- Good features to track: detection
		detector->detect(d_im0Gray, d_previousPoints);
		
		// --- Compute optical flow
		d_pyrLK->calc(d_im0Gray, d_im1Gray, d_previousPoints, d_nextPoints, d_status);

		// --- Swap old and new grey-scale images
		d_im0Gray = d_im1Gray;

		// --- Copies from device to host
		vector<Point2f> h_previousPoints(d_previousPoints.cols);
		device2Host(d_previousPoints, h_previousPoints);
		
		vector<Point2f> h_nextPoints(d_nextPoints.cols);
		device2Host(d_nextPoints, h_nextPoints);
		
		vector<uchar> status(d_status.cols);
		device2Host(d_status, status);
		
		// --- Draw optical flow
		drawFlow(im1, h_previousPoints, h_nextPoints, status, Scalar(255, 0, 0));
		imshow("PyrLK [Sparse]", im1);

		if (waitKey(10) > 0) break;
	}

}

