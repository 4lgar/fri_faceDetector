#ifndef FACES_H
#define FACES_H

//#define OUTPUT


#include <QtCore/QObject>
#include <QtCore/QList>

#include "opencv2/opencv.hpp"

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#define ORIGINAL_FAC 0.2
#define NEW_FAC 0.4

#include <cmath>

class Face
{
    public:
        Face();
        Face(int, int, float);
        Face(int, int, float, float);
        int x;
        int y;
        float rotation;
        float scale;

        static std::string ToString(QList<Face> *);
        static bool InitFaceDetection();
        static void DetectFace(cv::Mat *, QList<Face> *);
        static void RotateMat(cv::Mat *, cv::Mat *, int);
    private:
        void SetRotation(int x1, int y1, int x2, int y2);
};

#endif // FACES_H
