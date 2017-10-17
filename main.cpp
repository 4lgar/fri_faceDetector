#include <stdio.h>
#include <QCoreApplication>

#include "Face.h"

using namespace std;
using namespace cv;

int main() {
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat frame;

    QList<Face> faceDetected = QList<Face>();
    bool val = Face::InitFaceDetection();

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

