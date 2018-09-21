#if 0

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

#define max(a, b) (a > b ? a : b )
#define min(a, b) (a < b ? a : b )

//NOTE: These are the camera matrix and distortion coefficients for a Moto G phone.
//Will add a calibrator if I have the time.
double fCameraMatrix[3][3] = { { 1260.029, 0, 650.7481 },
										{ 0, 1259.2968, 341.3892 },
										{ 0, 0, 1 } };
//double fCameraMatrix[3][3] = { { 1265.8395, 0, 669.8386 },
//										{ 0, 1264.6891, 464.8123 },
//										{ 0, 0, 1 } };
cv::Mat mCameraMatrix = Mat( 3, 3, CV_64F, &fCameraMatrix );
double fDistortionCoeffs[] = { 0.1282, -0.646, 0, 0, 0.319 };
//double fDistortionCoeffs[] = { 0.0867, -0.3144, 0, 0, 0.1708 };
cv::Mat mDistortionCoeffs = Mat( 1, 5, CV_64F, &fDistortionCoeffs );

//TODO: Make these customizable in an XML or text file.
//All the measurements below are in millimeters, save for the pixel size which is in micrometers.
//Retrieved from http://www.ovt.com/products/sensor.php?id=43, which is the Pixy Camera's image sensor.
//The values underneath are actually for my phone (Moto G), changed for testing. Will change them back when done.
#define FOCAL_LENGTH 3.5f				//PIXY: 2.800f
//#define PIXEL_SIZE 1.4f					//PIXY: 3.0f	(micrometers)
#define IMAGE_SENSOR_W 3.6288f		//PIXY: 3.888f * 3.0f = 11.664f
//#define IMAGE_SENSOR_H 2.0384f				//PIXY: 2.430 * 3.0f = 7.290f

#define HUE_MIN 60	//COMP: ~160
#define HUE_MAX 100	//COMP: ~200
#define SAT_MIN 30
#define VAL_MIN 25

//Measured in centimeters.
//TODO: These are for an experimental target I used.
#define TARGET_WIDTH 4.15f
#define TARGET_HEIGHT 6.7f
#define TARGET_GROUND 26.0f		//Height from the ground

#define DOWNSCALE_AMOUNT 4

#define PI 3.1415926535897932384626433832795f

int TestAlpha( Mat &import );
int TestBravo( Mat &import );
int TestCharlie( Mat &import );
void TestDelta( void );
void TestEcho( void );

int nolongermain( int argc, char** argv )
{
	if(argc != 2)
    {
		cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
    }

	Mat import = imread(argv[1], IMREAD_COLOR);
	//TestEcho();
	return TestAlpha( import );
	//return TestCharlie( import );
	return 0;
}

//Testing the file system.
void TestEcho( void )
{
	cv::FileStorage fs( "_test.xml", cv::FileStorage::WRITE );
	fs << "test_int" << 5;

	int testInt;
	fs.open( "_test.xml", cv::FileStorage::READ );
	fs["test_int"] >> testInt;

	cout << testInt << endl;

	fs.release();
}

//Just testing video capture.
void TestDelta( void )
{
	cv::VideoCapture input(0);
	while ( true )
	{
		Mat frame;
		input >> frame;
		namedWindow( "Display window", WINDOW_AUTOSIZE );
		imshow( "Display window", frame );
		if(waitKey(15) >= 0) break;
	}
}


int TestCharlie( Mat &import )
{
	Mat image( import.rows, import.cols, CV_8U );
	cv::undistort( import, image, mCameraMatrix, mDistortionCoeffs );
	//cvtColor( image, image, CV_BGR2HSV );
	//cvtColor( import, import, CV_BGR2HSV );

	cout << image.rows << "\n";
	cout << image.cols << "\n";
	
    if( !image.data ) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

	//medianBlur( import, import, 9 );
	namedWindow( "Display window", WINDOW_AUTOSIZE );
    //imshow( "Display window", image );
	//waitKey(0);

	int curVal = 255;
	//cv::inRange( image, cv::Scalar( HUE_MIN / 2, (int)(SAT_MIN * 2.55), (int)(VAL_MIN * 2.55) ), cv::Scalar( HUE_MAX / 2, 255, 255 ), image );
	//while ( true )
	
		int rows = import.rows;
		int cols = import.cols;
		//if ( curVal <= 0 ) break;
		//curVal -= 5;
		//cv::resize( import, import, Size( cols / DOWNSCALE_AMOUNT, rows / DOWNSCALE_AMOUNT ), 0, 0, INTER_CUBIC );	//Downscale to speed up the processing
		//cv::resize( image, image, Size( cols / DOWNSCALE_AMOUNT, rows / DOWNSCALE_AMOUNT ), 0, 0, INTER_CUBIC );
		/*for (int i=0; i< image.rows; i++)
		{
			for (int j=0; j< image.cols; j++)
			{
				Vec3b* editValue = &import.at<Vec3b>(i,j);
				Vec3b* editValue2 = &image.at<Vec3b>(i,j);

				if ( (*editValue)[0] >= HUE_MIN / 2 && (*editValue)[0] <= HUE_MAX / 2
					&& (*editValue)[1] >= SAT_MIN * 2.55
					&& (*editValue)[2] >= curVal )
				{
					(*editValue2)[1] = 0;
					(*editValue2)[2] = 128;
				}
				else
				{
					(*editValue2) = 0;
					(*editValue2)[1] = 0;
					(*editValue2)[2] = 0;
				}
			}
		}*/
		//cv::cvtColor( image, image, CV_HSV2BGR );
		//cv::resize( import, import, Size( cols, rows ), 0, 0, INTER_CUBIC );	//Upscale
		//cv::resize( image, image, Size( cols, rows ), 0, 0, INTER_CUBIC );

		//imshow( "Display window", image );
		//waitKey(0);

		Mat canny_output;
		Canny( image, canny_output, 100, 200, 3 );
		vector<vector<Point> > contours_a;
		findContours( canny_output, contours_a, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
		Mat img2 = Mat::zeros( image.rows, image.cols, CV_8U );
		vector<vector<Point>> contours;
		for ( unsigned int i = 0; i < contours_a.capacity(); i++ )	//Purge contours that are too small.
		{
			if ( cv::contourArea( contours_a[i] ) < 10 ) continue;
			contours.push_back( contours_a[i] );
		}
		for ( unsigned int i = 0; i < contours.capacity(); i++ )
		{
			drawContours( image, contours, i, 255, 1, 8 );
			cv::rectangle( image, cv::boundingRect( contours[i] ), 128, 1 );
			cout << cv::boundingRect( contours[i] ) << endl;
		}

		//if ( contours.capacity() > 2 ) break;

		//cout << ".";
	

	cv::imwrite( "_temp.jpg", image );
	imshow( "Display window", canny_output );
	waitKey(0);

	return 0;
}

//Tests out merging contours into a single one, since the competition target'll have two strips of reflective tape on it.
//That way, we can use a RotatedRect's dimensions to give us more accurate calculations by considering the target's angle.
int TestBravo( Mat &import )
{
	Mat image( import.rows, import.cols, CV_8U );
	cv::undistort( import, image, mCameraMatrix, mDistortionCoeffs );
	cvtColor( image, image, CV_BGR2HSV );

	cout << image.rows << "\n";
	cout << image.cols << "\n";
	
    if( !image.data ) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

	medianBlur( image, image, 9 );
	//cv::inRange( image, cv::Scalar( HUE_MIN / 2, (int)(SAT_MIN * 2.55), (int)(VAL_MIN * 2.55) ), cv::Scalar( HUE_MAX / 2, 255, 255 ), image );
	for (int i=0; i< image.rows; i++)
    {
        for (int j=0; j< image.cols; j++)
        {
            Vec3b* editValue = &image.at<Vec3b>(i,j);

			if ( (*editValue)[0] >= HUE_MIN / 2 && (*editValue)[0] <= HUE_MAX / 2
				&& (*editValue)[1] >= SAT_MIN * 2.55
				&& (*editValue)[2] >= VAL_MIN * 2.55 )
			{
				(*editValue)[1] = 0;
				(*editValue)[2] = 128;
			}
			else
			{
				(*editValue) = 0;
				(*editValue)[1] = 0;
				(*editValue)[2] = 0;
			}
        }
    }
	cv::imwrite( "_temp.jpg", image );
	cv::cvtColor( image, image, CV_HSV2BGR );

	namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", image );

	waitKey(0);
	
	Mat canny_output;
	Canny( image, canny_output, 100, 200, 3 );
	vector<vector<Point> > contours_a;
	findContours( canny_output, contours_a, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
	Mat img2 = Mat::zeros( image.rows, image.cols, CV_8U );
	vector<vector<Point>> contours;
	for ( unsigned int i = 0; i < contours_a.capacity(); i++ )	//Purge contours that are too small.
	{
		if ( cv::contourArea( contours_a[i] ) < 10 ) continue;
		contours.push_back( contours_a[i] );
	}
	//assert( contours.capacity() == 2 );
	for ( unsigned int i = 0; i < contours.capacity(); i++ )
	{
		drawContours( img2, contours, i, 255, 1, 8 );
		//cv::rectangle( img2, cv::boundingRect( contours[i] ), 128, 1 );
		cout << cv::boundingRect( contours[i] ) << endl;
	}

	namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", img2 );

	waitKey(0);
	
	// = = = = RELEVANT CODE BEGINS HERE
	vector<Point> contour;
	std::vector<cv::Point> points;
    points.insert(points.end(), contours[0].begin(), contours[0].end());
    points.insert(points.end(), contours[1].begin(), contours[1].end());
	cv::approxPolyDP(cv::Mat(points), contour, 0.1, true);
	contours.insert( contours.end(), contour );
	drawContours( img2, contours, contours.capacity() - 1, 255 );

	RotatedRect d = cv::minAreaRect( contour );
	Point2f rect_points[4]; d.points( rect_points );
    for( int j = 0; j < 4; j++ )
		line( img2, rect_points[j], rect_points[(j+1)%4], 255, 1, 8 );
	// = = = = RELEVANT CODE ENDS HERE

    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", img2 );

	waitKey(0);

	return 0;
}

//Tests out range and angle detection with still images.
//Finding the x-distance from a rectangle's center to the middle of the image isn't working, gotta fix this.
int TestAlpha( Mat &import )
{
	Mat image( import.rows, import.cols, CV_8U );
	Mat image2( import.rows, import.cols, CV_8U );
	cv::undistort( import, image2, mCameraMatrix, mDistortionCoeffs );
	cvtColor( image2, image2, CV_BGR2HSV );

	cout << image.rows << "\n";
	cout << image.cols << "\n";
	
    if( !image.data ) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

	int cols = import.cols;
	int rows = import.rows;
	//cv::resize( import, import, Size( cols / DOWNSCALE_AMOUNT, rows / DOWNSCALE_AMOUNT ) );	//Downscale to speed up the processing
	//cv::resize( image, image, Size( cols / DOWNSCALE_AMOUNT, rows / DOWNSCALE_AMOUNT ) );

	//Filter out anything that is not greenish-blue (target's color)
	//Then make whatever's left plain white and convert back to BGR.
	//cv::inRange( image, cv::Scalar( HUE_MIN / 2, (int)(SAT_MIN * 2.55), (int)(VAL_MIN * 2.55) ), cv::Scalar( HUE_MAX / 2, 255, 255 ), image );
	for (int i=0; i< image.rows; i++)
    {
        for (int j=0; j< image.cols; j++)
        {
            Vec3b* editValue = &image2.at<Vec3b>(i,j);

			/*if ( (*editValue)[0] >= HUE_MIN / 2 && (*editValue)[0] <= HUE_MAX / 2
				&& (*editValue)[1] >= SAT_MIN * 2.55
				&& (*editValue)[2] >= VAL_MIN * 2.55 )
			{
				(*editValue)[1] = 0;
				(*editValue)[2] = 128;
			}
			else*/
			{
				//(*editValue)[2] = (*editValue)[2] * 0.2126f;
				//(*editValue)[1] = (*editValue)[1] * 0.7152f;
				//(*editValue)[0] = (*editValue)[0] * 0.0722f;
			}
        }
    }
	cv::medianBlur( image2, image2, 5 );
	threshold( image2, image, 30, 255, cv::THRESH_BINARY );
	imshow( "Display window", image );
	std::cout << ".\n";
    waitKey(0);
	threshold( image2, image, 35, 255, cv::THRESH_BINARY );
	imshow( "Display window", image );
	std::cout << ".\n";
    waitKey(0);
	threshold( image2, image, 40, 255, cv::THRESH_BINARY );
	imshow( "Display window", image );
	std::cout << ".\n";
    waitKey(0);
	threshold( image2, image, 45, 255, cv::THRESH_BINARY );
	imshow( "Display window", image );
	std::cout << ".\n";
    waitKey(0);
	threshold( image2, image, 50, 255, cv::THRESH_BINARY );
	imshow( "Display window", image );
	std::cout << ".\n";
    waitKey(0);
	threshold( image2, image, 55, 255, cv::THRESH_BINARY );
	imshow( "Display window", image );
	std::cout << ".\n";
    waitKey(0);
	threshold( image2, image, 60, 255, cv::THRESH_BINARY );
	imshow( "Display window", image );
	std::cout << ".\n";
    waitKey(0);
	//cv::resize( import, import, Size( cols, rows ) );	//Upscale
	//cv::resize( image, image, Size( cols, rows ) );
	//medianBlur( image, image, 5 );
	
	cvtColor( image, image, CV_HSV2BGR );
	cv::imwrite( "_temp.jpg", image );
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", image );
    waitKey(0);

	Mat canny_output;
	Canny( image, canny_output, 100, 200, 3 );
	vector<vector<Point> > contours_a;
	findContours( canny_output, contours_a, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
	Mat img2 = Mat::zeros( image.rows, image.cols, CV_8U );
	vector<vector<Point>> contours;
	//for ( unsigned int i = 0; i < contours_a.capacity(); i++ )	//Purge contours that are too small.
	//{
		//TODO: Replace the following line with something to get the largest two contours. Then we'll all be happy.
	//	if ( cv::contourArea( contours_a[i] ) < 50 ) continue;
	//	contours.push_back( contours_a[i] );
	//}
	//assert( contours.capacity() == 2 );
	for ( unsigned int i = 0; i < contours_a.capacity(); i++ )
	{
		drawContours( image, contours_a, i, 255, 1, 8 );
		//cv::rectangle( image, cv::boundingRect( contours[i] ), 128, 1 );
		cout << cv::boundingRect( contours_a[i] ) << endl;
		std::cout << i << " " << contours_a.capacity() << std::endl;
	}
	imshow( "Display window", image );
    waitKey(0);
	/*Rect a = cv::boundingRect( contours[0] );
	Rect b = cv::boundingRect( contours[1] );
	Rect c = Rect( cv::Point(min(a.x, b.x), min(a.y,b.y)), cv::Point(max(a.x+a.width,b.x+b.width), max(a.y+a.height,b.y+b.height)) );*/
	//assert( contours.capacity() == 1 );
	
	Rect c = cv::boundingRect( contours[0] );
	cout << c << endl;
	cv::rectangle( image, c, 255, 1 );
	RotatedRect d = cv::minAreaRect( contours[0] );
	cout << d.angle << endl;
	cout << d.size << endl;
	Point2f rect_points[4]; d.points( rect_points );
       for( int j = 0; j < 4; j++ )
          line( image, rect_points[j], rect_points[(j+1)%4], 255, 1, 8 );
	
	//If the picture's width:height ratio is over 10% lower than that of the real target,
	//we should assume that the target in the image is being cut off.
	//assert( (c.width / (float)c.height) >=
	//			(TARGET_WIDTH / TARGET_HEIGHT) - ((TARGET_WIDTH / TARGET_HEIGHT) * 0.1f) );

	//Here's another idea for finding the distance:
	/*
		Distance(mm) = f(mm) * real_height(mm) * image_height(pixels)
							-------------------------------------------------------
							object_height(pixels) * sensor_height(mm)
							
		To calculate theoretical height: H2 = (W2 * H1) / W1
		
		Also works with width instead of height.
		Using width, should work regardless of the vertical angle the target's viewed at.
		Only the object height/width would change, in the context of this program.
		Source: http://photo.stackexchange.com/questions/12434/how-do-i-calculate-the-distance-of-an-object-in-a-photo
	*/

	float fWidth = ( d.angle > -45.0f ? d.size.width : d.size.height );
	const float fApproxDistance = ( (FOCAL_LENGTH * (TARGET_WIDTH * 10.0f) * import.cols)
										/ (fWidth * IMAGE_SENSOR_W) ) / 10.0f;

	//Horizontal FOV = 2 * atan(d / 2f) [where d = sensor width and f = focal length)
	const float fApproxHFOV = (float)(2 * std::atan( IMAGE_SENSOR_W / (2 * FOCAL_LENGTH) ) * (180 / PI));

	int iXDist = std::abs( (640 - (c.x + c.width / 2)) );
	const float fApproxHAngle = (iXDist / ((float)image.cols / 2)) * fApproxHFOV;

	const float fApproxXDist = std::sqrt( fApproxDistance * fApproxDistance - TARGET_GROUND * TARGET_GROUND );	//a^2 + b^2 = c^2
	std::cout << TARGET_GROUND << endl;
	std::cout << fApproxDistance << endl;
	std::cout << TARGET_GROUND / fApproxDistance << endl;
	const float fApproxVAngle = std::asin( TARGET_GROUND / fApproxDistance ) * (180 / PI);	//sin = opp / hyp

	std::cout << "Approximate distance: " << fApproxDistance << "cm" << std::endl;
	std::cout << "Approximate horizontal angle: " << fApproxHAngle << " degrees [" << (c.x > image.cols / 2 ? "right" : "left") << "]" << std::endl;
	std::cout << "Approximate vertical angle: " << fApproxVAngle << " degrees" << std::endl;

	cv::imwrite( "_temp.jpg", image );
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", image );

    waitKey(0);
    return 0;
}

#endif