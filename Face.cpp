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

Face::Face(int x, int y, float scale, float rotation){
    this->x = x;
    this->y = y;
    this->rotation = rotation;
    this->scale = scale;
}

static CascadeClassifier faceCascade = CascadeClassifier();
static CascadeClassifier profileCascade = CascadeClassifier();
static CascadeClassifier eyeCascade = CascadeClassifier();
static CascadeClassifier eyeGlassCascade = CascadeClassifier();
static CascadeClassifier leftEyeCascade = CascadeClassifier();
static CascadeClassifier rightEyeCascade = CascadeClassifier();
static CascadeClassifier mouthCascade = CascadeClassifier();
static CascadeClassifier noseCascade = CascadeClassifier();

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
    return faceCascade.load("/home/qoqa/fri_faceDetector/haarcascades/haarcascade_frontalface_alt.xml") &&
           profileCascade.load("/home/qoqa/fri_faceDetector/haarcascades/haarcascade_profileface_alt.xml") &&
           eyeCascade.load("/home/qoqa/fri_faceDetector/haarcascades/haarcascade_eye.xml") &&
           eyeGlassCascade.load("/home/qoqa/fri_faceDetector/haarcascades/haarcascade_eye_tree_eyeglasses.xml") &&
           mouthCascade.load("/home/qoqa/fri_faceDetector/haarcascades/haarcascade_mouth.xml") &&
           leftEyeCascade.load("/home/qoqa/fri_faceDetector/haarcascades/haarcascade_lefteye.xml") &&
           rightEyeCascade.load("/home/qoqa/fri_faceDetector/haarcascades/haarcascade_righteye.xml") &&
           noseCascade.load("/home/qoqa/fri_faceDetector/haarcascades/haarcascade_nose.xml");
}

void Face::SetRotation(int x1, int y1, int x2, int y2){

    float opp = y2 - y1;
    float adj = x2 - x1;

    rotation = atan(opp/adj) * 180 / M_PI;
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
    vector<Rect> profileFaces;
//    vector<Rect> faces20;
//    vector<Rect> facesMin20;

    Mat frameGray;
//    Mat frame20;
//    Mat frameMin20;

    float scale = 0;

    Face toPush;

    cvtColor(*frame, frameGray, CV_BGR2GRAY );
    equalizeHist(frameGray, frameGray);

    // RotateMat(&frameGray, &frame20, 20);
    // RotateMat(&frameGray, &frameMin20, -20);

    //-- Detect faces
    faceCascade.detectMultiScale(frameGray, faces, 1.3, 3, 0| CV_HAAR_SCALE_IMAGE, Size(10, 10));
    profileCascade.detectMultiScale(frameGray, profileFaces, 1.3, 3, 0| CV_HAAR_SCALE_IMAGE, Size(10, 10));
    for (uint i = 0; i < profileFaces.size(); i++){
        profileFaces[i].x = profileFaces[i].x - profileFaces[i].width * 1/3;
        if (profileFaces[i].x + profileFaces[i].width >= frameGray.cols){
            profileFaces[i].x = frameGray.cols - profileFaces[i].width - 1;
        }
    }
    for (int i = profileFaces.size() - 1; i > 0; i--){
        for (uint j = 0; j < faces.size(); j++){
            Rect pf = profileFaces[i];
            Rect ff = faces[j];
            Rect in = pf & ff;
            if (in.width * in.height <= 0.5 * ff.width * ff.height) {
                faces.push_back(pf);
            }
        }
    }
    // faces.insert(faces.end(), profileFaces.begin(), profileFaces.end());

    Mat flipped;
    flip(frameGray, flipped, 1);
    profileCascade.detectMultiScale(flipped, profileFaces, 1.3, 3, 0| CV_HAAR_SCALE_IMAGE, Size(10, 10));
    for (int i = profileFaces.size() - 1; i > 0; i--){
        for (uint j = 0; j < faces.size(); j++){
            Rect pf = profileFaces[i];
            Rect ff = faces[j];
            Rect in = pf & ff;
            if (in.width * in.height <= 0.5 * ff.width * ff.height) {
                faces.push_back(pf);
            }
        }
    }
//    faces.insert(faces.end(), profileFaces.begin(), profileFaces.end());


    toReturn->clear();

    for(uint i = 0; i < faces.size(); ++i){
        Rect face = faces[i];

        scale = (float)faces[i].height / (float)frame->rows;

        // Try to detect eyes
        Rect roi(face.x, face.y, face.width, face.height);
        Rect roiRightEye(face.x + face.width * 1/3, face.y, face.width * 2/3, face.height * 2/3);
        Rect roiLeftEye(face.x, face.y, face.width * 2/3, face.height * 2/3);
        Rect roiMouth(face.x, face.y + face.height * 1/2, face.width, face.height * 1/2);


        vector<Rect>rightEyes;
        vector<Rect>leftEyes;
        vector<Rect>mouths;
//        vector<Rect>noses;

        rightEyeCascade.detectMultiScale(frameGray(roiRightEye), rightEyes, 1.3, 2, 0 | CV_HAAR_SCALE_IMAGE);
        leftEyeCascade.detectMultiScale(frameGray(roiLeftEye), leftEyes, 1.3, 5, 0 | CV_HAAR_SCALE_IMAGE);
        mouthCascade.detectMultiScale(frameGray(roiMouth), mouths, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(face.width * 0.1, face.width * 0.1),  Size(face.width * 0.8, face.width * 0.8));
//        noseCascade.detectMultiScale(frameGray(roi), noses, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(face.width * 0.1, face.width * 0.1),  Size(face.width * 0.8, face.width * 0.8));


        Mat output;
        frameGray(roi).copyTo(output);

        int biggestIndex = -1;
        float biggest = 0;

        for (uint j = 0; j < mouths.size(); j++){
            if (mouths[j].width > biggest)
                biggestIndex = j;

        }
        Rect mouth;
        if (biggestIndex != -1){
            mouth = mouths[biggestIndex];
            mouth.y += face.height * 1/2;
            rectangle(output, mouth, Scalar(255, 0, 0));
        }
//        for (uint j = 0; j < noses.size(); j++){
//            rectangle(output, noses[j], Scalar(128, 0, 0));
//        }
        int rightMostIndex = -1;
        float rightMost = 0;
        for (uint j = 0; j < rightEyes.size(); j++){
            if (rightEyes[j].x > rightMost)
                rightMostIndex = j;
        }
        Rect rightEye;
        if (rightMostIndex != -1){
            rightEye = rightEyes[rightMostIndex];
            rightEye.x += face.width * 1/3;
            rectangle(output, rightEye, Scalar(200, 0, 0));
        }

        int leftMostIndex = -1;
        float leftMost = 99999999;
        for (uint j = 0; j < leftEyes.size(); j++){
            if (leftEyes[j].x < leftMost)
                leftMostIndex = j;
        }
        Rect leftEye;
        if (leftMostIndex != -1){
            leftEye = leftEyes[leftMostIndex];
            rectangle(output, leftEye, Scalar(200, 0, 0));
        }

        float countAngles = 0;
        float angleSum = 0;
        if (leftMostIndex != -1 && rightMostIndex != -1){
            countAngles++;
            Point centerL(leftEye.x + leftEye.width * 1/2, leftEye.y + leftEye.height * 1/2);
            Point centerR(rightEye.x + rightEye.width * 1/2, rightEye.y + rightEye.height * 1/2);
            line(output, centerL, centerR, Scalar(200));
            float angle = atan2(centerR.y - centerL.y, centerR.x - centerL.x);
            angleSum += angle * 180 / M_PI;
        }

        if (leftMostIndex != -1 && biggestIndex != -1){
            countAngles++;
            Point centerE(leftEye.x + leftEye.width * 1/2, leftEye.y + leftEye.height * 1/2);
            Point centerM(mouth.x + mouth.width * 1/2, mouth.y + mouth.height * 1/2);
            line(output, centerE, centerM, Scalar(200));
            float angle = atan2(centerM.y - centerE.y, centerM.x - centerE.x);
            angleSum += angle * 180 / M_PI - 67;
        }

        if (rightMostIndex != -1 && biggestIndex != -1){
            countAngles++;
            Point centerE(rightEye.x + rightEye.width * 1/2, rightEye.y + rightEye.height * 1/2);
            Point centerM(mouth.x + mouth.width * 1/2, mouth.y + mouth.height * 1/2);
            line(output, centerE, centerM, Scalar(200));
            float angle = atan2(centerM.y - centerE.y, centerM.x - centerE.x);
            angleSum += angle * 180 / M_PI - 114;
        }

        float finalAngle = 0;
        if (countAngles > 0)
            finalAngle = angleSum / countAngles;

//        imshow("Face", output);

        toPush = Face(faces[i].x, faces[i].y, scale, finalAngle);
        toReturn->push_back(toPush);

    }

//    for(uint i = 0; i < faces20.size(); ++i){

//        scale = (float)faces20[i].height / (float)frame->rows;
//        toPush = Face(faces20[i].x, faces20[i].y, scale);
//        toPush.rotation = 20;
//        toReturn->push_back(toPush);

//    }

//    for(uint i = 0; i < facesMin20.size(); ++i){

//        scale = (float)facesMin20[i].height / (float)frame->rows;
//        toPush = Face(facesMin20[i].x, facesMin20[i].y, scale);
//        toPush.rotation = -20;
//        toReturn->push_back(toPush);

//    }

}
