#include "Face.h"

using namespace std;
using namespace cv;

Face::Face(int x, int y, int rotation){
    this->x = x;
    this->y = y;
    this->rotation = rotation;
    scale = 1;
}

static CascadeClassifier faceCascade = CascadeClassifier();

std::string Face::ToString(QList<Face> *list){

    QString *toReturn = new QString();

    for(int i = 0; i < list->length(); i++){

        toReturn->append(
            QString("%1 %2 %3 %4").arg(QString::number(list->at(i).x), QString::number(list->at(i).y), QString::number(list->at(i).rotation), QString::number(list->at(i).scale))
        );

        if(i != list->length() - 1)
            toReturn->append(" ");
    }

    return toReturn->toUtf8().constData();

}

bool Face::InitFaceDetection(){
    return faceCascade.load("/home/dons/Téléchargements/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml");
}

void Face::DetectFace(Mat *frame, QList<Face> *toReturn){

    vector<Rect> faces;
    Mat frameGray;

    cvtColor(*frame, frameGray, CV_BGR2GRAY );
    equalizeHist(frameGray, frameGray);

    //-- Detect faces
    faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    toReturn->clear();

    for(uint i = 0; i < faces.size(); ++i){
        toReturn->push_back(Face(faces[i].x, faces[i].y, 10));
    }

}
