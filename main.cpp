#include <stdio.h>

#include "Face.h"

#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

inline bool isInteger(const std::string & s)
{
   if(s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false ;

   char * p ;
   strtol(s.c_str(), &p, 10) ;

   return (*p == 0) ;
}

int main(int argc, char *argv[]) {
    #ifdef OUTPUT
    namedWindow("Face", WINDOW_NORMAL);
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

            if( !frame.empty() ){
                Mat smallerFrame;
                cv::resize(frame, smallerFrame, Size(), NEW_FAC, NEW_FAC, INTER_AREA);

//                imshow("Face", smallerFrame);

                Face::DetectFace(&smallerFrame, &faceDetected);
                if(faceDetected.count() > 0)
                    std::cout << Face::ToString(&faceDetected) << std::endl;
                else
                    std::cout << std::endl;

            }
#ifdef OUTPUT
            waitKey(1);
#endif
        }

        // the camera will be deinitialized automatically in VideoCapture destructor
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

