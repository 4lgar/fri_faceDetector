#include <stdio.h>
#include <iostream>
#include <string>

#include "Face.h"

#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

#define ORIGINAL_FAC 0.2
#define NEW_FAC 0.6

inline bool isInteger(const std::string & s)
{
   if(s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false ;

   char * p ;
   strtol(s.c_str(), &p, 10) ;

   return (*p == 0) ;
}

int main(int argc, char *argv[]) {
    #ifdef OUTPUT
    // namedWindow("Face", WINDOW_NORMAL);
    namedWindow("Results", WINDOW_OPENGL);
    #endif

    if(argc != 2){

        std::cout << "Wrong argument count. Specify a capture device ID or jpeg file." << std::endl;
        return -1;
    }

    string argument = argv[1];
    if (isInteger(argument)){
        VideoCapture cap(atoi(argv[1])); // open the default camera
        if(!cap.isOpened())  // check if we succeeded
            return -1;

        Mat frame;

        QList<Face> faceDetected = QList<Face>();

        if(!Face::InitFaceDetection()){
            std::cout << "Could not load face detection libraries" << endl;
            return -1;
        }

        for(;;)
        {
            cap >> frame;

            if (frame.cols == 0 || frame.rows == 0) {
                cap.release();
                cap.open(atoi(argv[1]));
                continue;
            }

            Mat smallerFrame;

            if( !frame.empty() ){
                cv::resize(frame, smallerFrame, Size(), NEW_FAC, NEW_FAC, INTER_AREA);

                //frame.copyTo(smallerFrame);
//                imshow("Face", smallerFrame);

                Face::DetectFace(&smallerFrame, &faceDetected);
                if(faceDetected.count() > 0)
                    std::cout << Face::ToString(&faceDetected) << std::endl;
                else
                    std::cout << std::endl;

            }
#ifdef OUTPUT
            waitKey(1);
            for (int i = 0; i < faceDetected.length(); i++){
                rectangle(smallerFrame,
                          Point(faceDetected[i].x, faceDetected[i].y),
                          Point(faceDetected[i].x + faceDetected[i].scale, faceDetected[i].y + faceDetected[i].scale),
                          Scalar(255, 0, 255));
            }
            putText(smallerFrame, to_string(faceDetected.length()), Point(90, 200), FONT_HERSHEY_SIMPLEX, 8, Scalar(255, 255, 0), 3);
            imshow("Results", smallerFrame);
#endif
        }

        // the camera will be dei nitialized automatically in VideoCapture destructor
        return 0;
    } else {

        Mat frame = imread(argument, IMREAD_COLOR);

        if(frame.empty())
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }

        QList<Face> faceDetected = QList<Face>();

        if(!Face::InitFaceDetection()){
            std::cout << "Could not load face detection libraries" << endl;
            return -1;
        }

        Face::DetectFace(&frame, &faceDetected);

        if(faceDetected.count() > 0)
            std::cout << Face::ToString(&faceDetected) << std::endl;

        return 0;
    }
}

