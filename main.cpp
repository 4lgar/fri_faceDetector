#include <stdio.h>
#include <QCoreApplication>
#include <QFile>

#include "Face.h"

#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    if(argc == 2){

        Mat frame = imread(argv[1], IMREAD_COLOR);

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

    }else{

        VideoCapture cap(0); // open the default camera
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

                Face::DetectFace(&frame, &faceDetected);
                if(faceDetected.count() > 0)
                    std::cout << Face::ToString(&faceDetected) << std::endl;

            }
        }

        // the camera will be deinitialized automatically in VideoCapture destructor
        return 0;

    }
}

