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

void Face::RotateMat(Mat *src, Mat *dst, int angle){
    // get rotation matrix for rotating the image around its center
    Point2f center(src->cols/2.0, src->rows/2.0);
    Mat rot = getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle
    Rect bbox = RotatedRect(center,src->size(), angle).boundingRect();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - center.x;
    rot.at<double>(1,2) += bbox.height/2.0 - center.y;

    warpAffine(*src, *dst, rot, bbox.size());
}

void Face::DetectFace(Mat *frame, QList<Face> *toReturn){

    vector<Rect> faces;
    vector<Rect> faces20;
    vector<Rect> facesMin20;

    Mat frameGray;
    Mat frame20;
    Mat frameMin20;

    float scale = 0;

    Face toPush;

    cvtColor(*frame, frameGray, CV_BGR2GRAY );
    equalizeHist(frameGray, frameGray);

    RotateMat(&frameGray, &frame20, 20);
    RotateMat(&frameGray, &frameMin20, -20);

    //-- Detect faces
    faceCascade.detectMultiScale(frameGray, faces, 1.3, 2, 0| CV_HAAR_SCALE_IMAGE, Size(30, 30));
    faceCascade.detectMultiScale(frame20, faces20, 1.3, 2, 0| CV_HAAR_SCALE_IMAGE, Size(30, 30));
    faceCascade.detectMultiScale(frameMin20, facesMin20, 1.3, 2, 0| CV_HAAR_SCALE_IMAGE, Size(30, 30));


    toReturn->clear();

    for(uint i = 0; i < faces.size(); ++i){

        scale = (float)faces[i].height / (float)frame->rows;
        toPush = Face(faces[i].x, faces[i].y, scale);
        toReturn->push_back(toPush);

    }

    for(uint i = 0; i < faces20.size(); ++i){

        scale = (float)faces20[i].height / (float)frame->rows;
        toPush = Face(faces20[i].x, faces20[i].y, scale);
        toPush.rotation = 20;
        toReturn->push_back(toPush);

    }

    for(uint i = 0; i < facesMin20.size(); ++i){

        scale = (float)facesMin20[i].height / (float)frame->rows;
        toPush = Face(facesMin20[i].x, facesMin20[i].y, scale);
        toPush.rotation = -20;
        toReturn->push_back(toPush);

    }

}
