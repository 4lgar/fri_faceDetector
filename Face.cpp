#include "Face.h"

using namespace std;
using namespace cv;

Face::Face(){
    this->x = 0;
    this->y = 0;
    this->rotation = 0;
    this->scale = 1;
}

Face::Face(int x, int y, float scale){
    this->x = x;
    this->y = y;
    this->rotation = 0;
    this->scale = scale;
}

static CascadeClassifier faceCascade = CascadeClassifier();
static CascadeClassifier eyeCascade = CascadeClassifier();
static CascadeClassifier eyeGlassCascade = CascadeClassifier();

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
    return faceCascade.load("/home/dons/Téléchargements/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml") &&
           eyeCascade.load("/home/dons/Téléchargements/opencv-3.3.0/data/haarcascades/haarcascade_eye.xml") &&
           eyeGlassCascade.load("/home/dons/Téléchargements/opencv-3.3.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
}

void Face::SetRotation(int x1, int y1, int x2, int y2){

    float opp = y2 - y1;
    float adj = x2 - x1;

    rotation = qRadiansToDegrees(qAtan(opp/adj));
}

void Face::DetectFace(Mat *frame, QList<Face> *toReturn){

    vector<Rect> faces;
    std::vector<Rect> eyes;
    std::vector<Rect> eyesGlass;

    Mat frameGray;
    Mat faceROI;

    float scale = 0;

    Face toPush;

    cvtColor(*frame, frameGray, CV_BGR2GRAY );
    equalizeHist(frameGray, frameGray);

    //-- Detect faces
    faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0| CV_HAAR_SCALE_IMAGE, Size(30, 30));

    toReturn->clear();

    for(uint i = 0; i < faces.size(); ++i){

        scale = (float)faces[i].height / (float)frame->rows;

        toPush = Face(faces[i].x, faces[i].y, scale);

        faceROI = frameGray(faces[i]);

        if(!faceROI.empty()){
            eyeCascade.detectMultiScale(faceROI, eyes, 1.08, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
            eyeGlassCascade.detectMultiScale(faceROI, eyesGlass, 1.01, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

            if(eyesGlass.size() >= 2)
                toPush.SetRotation(eyesGlass[0].x, eyesGlass[0].y, eyesGlass[1].x, eyesGlass[1].y);

            if(eyes.size() >= 2)
                toPush.SetRotation(eyes[0].x, eyes[0].y, eyes[1].x, eyes[1].y);

        }

        toReturn->push_back(toPush);

        eyes.clear();
    }

}
