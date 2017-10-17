#ifndef FACES_H
#define FACES_H

#include <QtCore/QObject>
#include <QtCore/QList>

#include "opencv2/opencv.hpp"

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class Face
{
    public:
        Face(int, int, int);
        int x;
        int y;
        int rotation;
        float scale;

        static std::string ToString(QList<Face> *);
        static bool InitFaceDetection();
        static void DetectFace(cv::Mat *, QList<Face> *);
};

#endif // FACES_H
