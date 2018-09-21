#if 0

/*
	targetting.cpp
	The targetting program itself.

	Primarily written by Gabriel Valachi.
	If you need immediate assistance, ask McDonough for my cell phone number.
	For all else, you already have my email address.
	Please highlight any and all modifications.
*/

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>

//#define RASPBERRY_PI	//Uncomment when compiling on the Raspberry Pi.
#ifdef RASPBERRY_PI	//Needed for UART output to communicate with the roboRIO.
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#endif

#define max(a, b) (a > b ? a : b )
#define min(a, b) (a < b ? a : b )
#define PI 3.1415926535897932384626433832795f

//Options filename.
#define OPS_FILE "_options.xml"

//Default values for when OPS_FILE can't be found.
//Hue is between 0 and 180, and saturation and value are between 0 and 255.
//I just set them this way according to my readings via Paint.NET.
#define HUE_MIN_DFLT 160
#define HUE_MAX_DFLT 200
#define SAT_MIN_DFLT 75
#define VAL_MIN_DFLT 25

//How much to downscale an image by in fast mode.
//Don't change this, I only used a preprocessor definition for testing.
#define DOWNSCALE_AMOUNT 4

//Default camera properties. These are for the Pixy Camera that apparently we'll use.
//Measurements in millimeters.
#define FOCAL_LENGTH_DFLT 2.800f
#define IMAGE_SENSOR_W_DFLT 11.664f
#define IMAGE_SENSOR_H_DFLT 7.290f

//Default target measurements in centimeters.
//I put zero for them because I don't have the manual with me.
#define TARGET_WIDTH_DFLT 0
#define TARGET_GROUND_DFLT 0
#define TARGET_CAMERA_OFFSET_DFLT 0	//Camera's distance from the ground.

//Default camera and distortion matrices of coefficients.
double CAMERA_MATRIX_DFLT_A[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 1 } };	//3x3
cv::Mat CAMERA_MATRIX_DFLT = cv::Mat( 3, 3, CV_64F, &CAMERA_MATRIX_DFLT );
double DISTORTION_COEFFS_DFLT_A[5] = { 0, 0, 0, 0, 0 };	//1x5
cv::Mat DISTORTION_COEFFS_DFLT = cv::Mat( 1, 5, CV_64F, &DISTORTION_COEFFS_DFLT );

//Function prototypes.
void FilterTargets( cv::Mat &import, cv::Mat &image, int iHueMin, int iHueMax, int iSatMin, int iValMin );
std::vector<cv::Point> GrabLargestContour( std::vector<std::vector<cv::Point>> vContours, int &index );

int currentlynotmain( int argc, char** argv )
{
#ifdef RASPBERRY_PI	//Setup code copied straight from http://www.raspberry-projects.com/pi/programming-in-c/uart-serial-port/using-the-uart.
	//-------------------------
	//----- SETUP USART 0 -----
	//-------------------------
	//At bootup, pins 8 and 10 are already set to UART0_TXD, UART0_RXD (ie the alt0 function) respectively
	int uart0_filestream = -1;
	
	//OPEN THE UART
	//The flags (defined in fcntl.h):
	//	Access modes (use 1 of these):
	//		O_RDONLY - Open for reading only.
	//		O_RDWR - Open for reading and writing.
	//		O_WRONLY - Open for writing only.
	//
	//	O_NDELAY / O_NONBLOCK (same function) - Enables nonblocking mode. When set read requests on the file can return immediately with a failure status
	//											if there is no input immediately available (instead of blocking). Likewise, write requests can also return
	//											immediately with a failure status if the output can't be written immediately.
	//
	//	O_NOCTTY - When set and path identifies a terminal device, open() shall not cause the terminal device to become the controlling terminal for the process.
	uart0_filestream = open("/dev/ttyAMA0", O_RDWR | O_NOCTTY | O_NDELAY);		//Open in non blocking read/write mode
	if (uart0_filestream == -1)
	{
		//ERROR - CAN'T OPEN SERIAL PORT
		printf("Error - Unable to open UART.  Ensure it is not in use by another application\n");
	}
	
	//CONFIGURE THE UART
	//The flags (defined in /usr/include/termios.h - see http://pubs.opengroup.org/onlinepubs/007908799/xsh/termios.h.html):
	//	Baud rate:- B1200, B2400, B4800, B9600, B19200, B38400, B57600, B115200, B230400, B460800, B500000, B576000, B921600, B1000000, B1152000, B1500000, B2000000, B2500000, B3000000, B3500000, B4000000
	//	CSIZE:- CS5, CS6, CS7, CS8
	//	CLOCAL - Ignore modem status lines
	//	CREAD - Enable receiver
	//	IGNPAR = Ignore characters with parity errors
	//	ICRNL - Map CR to NL on input (Use for ASCII comms where you want to auto correct end of line characters - don't use for bianry comms!)
	//	PARENB - Parity enable
	//	PARODD - Odd parity (else even)
	struct termios options;
	tcgetattr(uart0_filestream, &options);
	options.c_cflag = B9600 | CS8 | CLOCAL | CREAD;		//<Set baud rate
	options.c_iflag = IGNPAR;
	options.c_oflag = 0;
	options.c_lflag = 0;
	tcflush(uart0_filestream, TCIFLUSH);
	tcsetattr(uart0_filestream, TCSANOW, &options);
#endif

	int iHueMin, iHueMax, iSatMin, iValMin;
	float fFocalLength, fImageSensorW;
	float fTargetWidth, fTargetGround, fTargetCameraOffset;
	cv::Mat mCameraMatrix, mDistortionCoeffs;
	bool bFast;	//Whether or not to downscale the image.
	
	cv::FileStorage fs( OPS_FILE, cv::FileStorage::READ );
	if ( fs.isOpened() )
	{
		fs["hue_min"] >> iHueMin;
		fs["hue_max"] >> iHueMax;
		fs["sat_min"] >> iSatMin;
		fs["val_min"] >> iValMin;

		fs["cam_focal_length"] >> fFocalLength;
		fs["cam_image_sensor_width"] >> fImageSensorW;

		fs["target_width"] >> fTargetWidth;
		fs["target_ground"] >> fTargetGround;
		fs["target_camera_offset"] >> fTargetCameraOffset;

		fs["cam_matrix"] >> mCameraMatrix;
		fs["cam_distortion_coeffs"] >> mDistortionCoeffs;

		fs["start_fast"] >> bFast;
		
		fs.release();
	}
	else	//Set default values for above declared variables, and create a new XML file.
	{
		std::cout << "!!! Error reading " << OPS_FILE << ". Creating new file with default parameters. !!!\n";
		cv::FileStorage fs2( OPS_FILE, cv::FileStorage::WRITE );
		fs2 << "hue_min" << HUE_MIN_DFLT;
		fs2 << "hue_max" << HUE_MAX_DFLT;
		fs2 << "sat_min" << SAT_MIN_DFLT;
		fs2 << "val_min" << VAL_MIN_DFLT;
		iHueMin = HUE_MIN_DFLT;
		iHueMax = HUE_MAX_DFLT;
		iSatMin = SAT_MIN_DFLT;
		iValMin = VAL_MIN_DFLT;

		fs2 << "cam_focal_length" << FOCAL_LENGTH_DFLT;
		fs2 << "cam_image_sensor_width" << IMAGE_SENSOR_W_DFLT;
		fFocalLength = FOCAL_LENGTH_DFLT;
		fImageSensorW = IMAGE_SENSOR_W_DFLT;

		fs2 << "target_width" << TARGET_WIDTH_DFLT;
		fs2 << "target_ground" << TARGET_GROUND_DFLT;
		fs2 << "target_camera_offset" << TARGET_CAMERA_OFFSET_DFLT;
		fTargetWidth = TARGET_WIDTH_DFLT;
		fTargetGround = TARGET_GROUND_DFLT;
		fTargetCameraOffset = TARGET_CAMERA_OFFSET_DFLT;

		fs2 << "cam_matrix" << CAMERA_MATRIX_DFLT;
		fs2 << "cam_distortion_coeffs" << DISTORTION_COEFFS_DFLT;
		mCameraMatrix = CAMERA_MATRIX_DFLT;
		mDistortionCoeffs = DISTORTION_COEFFS_DFLT;

		fs2 << "start_fast" << false;
		bFast = false;

		fs2.release();
	}

	//TODO: Add a calibrator for if the camera hasn't been calibrated.
	
	cv::VideoCapture cap(0);
	cv::Mat mCurFrame;
	int cols, rows;
	while ( true )
	{
		cap >> mCurFrame;
		cols = mCurFrame.cols;
		rows = mCurFrame.rows;
		cv::imwrite( "_temp.jpg", mCurFrame );

		cv::Mat image( mCurFrame.rows, mCurFrame.cols, CV_8U );
		//cv::undistort( mCurFrame, image, mCameraMatrix, mDistortionCoeffs );
		image = mCurFrame.clone();
		// ^ Replace with image = mCurFrame.clone(); if not using it.
		if ( bFast )	//Downscale in fast mode
		{
			cv::resize( mCurFrame, mCurFrame, cv::Size( cols / DOWNSCALE_AMOUNT, rows / DOWNSCALE_AMOUNT ) );
			cv::resize( image, image, cv::Size( cols / DOWNSCALE_AMOUNT, rows / DOWNSCALE_AMOUNT ) );
		}
		FilterTargets( mCurFrame, image, iHueMin, iHueMax, iSatMin, iValMin );
		if ( bFast )
		{
			cv::resize( mCurFrame, mCurFrame, cv::Size( cols, rows ) );
			cv::resize( image, image, cv::Size( cols, rows ) );
		}
		if ( !bFast ) cv::medianBlur( image, image, 9 );	//Be warned that this causes performance drops.
		
		//Do an edge detection and find the contours in the image.
		cv::Mat contours_detected;
		cv::Canny( image, contours_detected, 100, 200 );
		std::vector<std::vector<cv::Point>> contours_unsorted;
		cv::findContours( contours_detected, contours_unsorted, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
		if ( contours_unsorted.capacity() < 2 )
		{
			cv::waitKey(30);
			continue;
		}
		
		//Grab the largest two contours.
		std::vector<std::vector<cv::Point>> contours;
		int largestIndex = 0;
		contours.push_back( GrabLargestContour( contours_unsorted, largestIndex ) );
		contours_unsorted.erase( contours_unsorted.begin() + largestIndex );
		contours.push_back( GrabLargestContour( contours_unsorted, largestIndex ) );
#ifndef RASPBERRY_PI
		cv::drawContours( image, contours, 0, cv::Scalar( 0, 0, 255 ) );
		cv::drawContours( image, contours, 1, cv::Scalar( 0, 255, 0 ) );
#endif
		//If the second largest contour's area is less than 10% in size, assume that it's just noise and skip.
		//Not recommended if you'll be using a noisy webcam.
		/*if ( cv::contourArea( contours[1] ) < (cv::contourArea( contours[0] ) * 0.1f) )
		{
			if ( cv::waitKey(30) > 0 ) break;
			continue;
		}*/

		//Connect the two contours together and treat them as one.
		std::vector<cv::Point> contour;
		std::vector<cv::Point> points;
		points.insert(points.end(), contours[0].begin(), contours[0].end());
		points.insert(points.end(), contours[1].begin(), contours[1].end());
		cv::approxPolyDP(cv::Mat(points), contour, 0.1, true);
		contours.insert( contours.end(), contour );
#ifndef RASPBERRY_PI
		drawContours( image, contours, contours.capacity() - 1, 255 );
#endif

		//Finally, get the approximate distances and angles.
		cv::RotatedRect rRotatedBoundingBox = cv::minAreaRect( contour );
#ifndef RASPBERRY_PI
		cv::Point2f rect_points[4]; rRotatedBoundingBox.points( rect_points );
		for( int j = 0; j < 4; j++ )
			line( image, rect_points[j], rect_points[(j+1)%4], 255, 1, 8 );
#endif

		float fTargetHeight = fTargetGround - fTargetCameraOffset;
		float fWidth = ( rRotatedBoundingBox.angle > -45.0f ? rRotatedBoundingBox.size.width : rRotatedBoundingBox.size.height );
		const float fApproxHypDist = ( (fFocalLength * (fTargetWidth * 10.0f) * image.cols )
										/ (fWidth * fImageSensorW) / 10.0f );	//Approximate distance in centimeters.

		float fApproxHFOV = (float)(2 * std::atan( fImageSensorW / (2 * fFocalLength) ) * (180 / PI));
		int iXDist = (640 - (rRotatedBoundingBox.boundingRect().x + rRotatedBoundingBox.boundingRect().width / 2));
		const float fApproxHAngle = (iXDist / ((float)image.cols / 2)) * fApproxHFOV;

		const float fApproxXDist = std::sqrt( fApproxHypDist * fApproxHypDist - fTargetHeight * fTargetHeight );
		const float fApproxVAngle = std::asin( fTargetHeight / fApproxHypDist ) * (180 / PI);

		std::cout << fApproxHypDist << std::endl;

		//TODO: Not sure if this works. You guys'll need to test it.
#ifdef RASPBERRY_PI	//Copied and modified from http://www.raspberry-projects.com/pi/programming-in-c/uart-serial-port/using-the-uart.
		unsigned float tx_buffer[4];
		unsigned char *p_tx_buffer;
	
		p_tx_buffer = &tx_buffer[0];
		*p_tx_buffer++ = fApproxHypDist;
		*p_tx_buffer++ = fApproxHAngle;
		*p_tx_buffer++ = fApproxXDist;
		*p_tx_buffer++ = fApproxVAngle;
	
		if (uart0_filestream != -1)
		{
			int count = write(uart0_filestream, &tx_buffer[0], (p_tx_buffer - &tx_buffer[0]));		//Filestream, bytes to write, number of bytes to write
			if (count < 0)
			{
				printf("UART TX error\n");
			}
		}

		//----- CHECK FOR ANY RX BYTES -----
		if (uart0_filestream != -1)
		{
			// Read up to 255 characters from the port if they are there
			unsigned char rx_buffer[256];
			int rx_length = read(uart0_filestream, (void*)rx_buffer, 255);		//Filestream, buffer to store in, number of bytes to read (max)
			if (rx_length < 0)
			{
				//An error occured (will occur if there are no bytes)
			}
			else if (rx_length == 0)
			{
				//No data waiting
			}
			else
			{
				//Bytes received
				bFast = !bFast;	//Switch between fast modes if we get input from the roboRIO.
			}
		}
#else
		cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
		cv::imshow( "Display window", image );
		if ( cv::waitKey(30) > 0 ) break;
#endif
	}

	return 0;
}

// Filters everything within the HSV limits out of the input image.
//TODO: Check for RGB values too to make sure something is ACTUALLY green!
void FilterTargets( cv::Mat &import, cv::Mat &image, int iHueMin, int iHueMax, int iSatMin, int iValMin )
{
	cv::cvtColor( image, image, CV_BGR2HSV );
	for ( int i = 0; i < image.rows; i++ )
	{
		for ( int j = 0; j < image.cols; j++ )
		{
			cv::Vec3b* hCurPixel = &image.at<cv::Vec3b>(i, j);
			if ( (*hCurPixel)[0] >= iHueMin / 2 && (*hCurPixel)[0] <= iHueMax / 2
					&& (*hCurPixel)[1] >= iSatMin * 2.55
					&& (*hCurPixel)[2] >= iValMin * 2.55 )
			{
				(*hCurPixel)[1] = 0;
				(*hCurPixel)[2] = 255;
			}
			else
			{
				(*hCurPixel)[0] = (*hCurPixel)[1] = (*hCurPixel)[2] = 0;
			}
		}
	}
	cv::cvtColor( image, image, CV_HSV2BGR );
}

//Returns the largest contour from the input vector.
std::vector<cv::Point> GrabLargestContour( std::vector<std::vector<cv::Point>> contours, int &index )
{
	std::vector<cv::Point> vLargestContour = contours[0];
	for ( unsigned int i = 0; i < contours.capacity(); i++ )
	{
		if ( cv::contourArea( contours[i] ) > cv::contourArea( vLargestContour ) )
		{
			vLargestContour = contours[i];
			index = i;
		}
	}
	
	return vLargestContour;
}
#endif