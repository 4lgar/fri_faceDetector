#ifndef FACES_H
#define FACES_H

//#define OUTPUT


#include <QtCore/QObject>
#include <QtCore/QList>

#include <opencv2/core.hpp>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/video/background_segm.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cmath>
#include <stdio.h>
#include <iostream>

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
