#include "Face.h"

#define OVERLAP_TOLERANCE 0.1

//#define SIDE_FACES
#define ROTATION

//#define BACKGROUND_SUBSTRACT
//#define EQUALIZE_HIST
#define TEMPLATE_MATCHING 200  // If this exists, then the value of this constant is the "confidence" needed
#define TEMPLATE_MATCHING_ROI_MARGIN 20
#define TEMPLATE_MATCHING_CLEANUP_INTERVAL 20

#define GPU

#define SIDE_FACE_SCALE_RIGHT 0.85
#define SIDE_FACE_SCALE_LEFT 0.75

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
#ifdef OUTPUT
    namedWindow("Debug", WINDOW_OPENGL);
#endif

    return
#ifdef LBP
            faceCascade.load("/home/fullrange/fri_faceDetector/lbpcascades/lbpcascade_frontalface_improved.xml") &&
            profileCascade.load("/home/fullrange/fri_faceDetector/lbpcascades/lbpcascade_profileface.xml") &&
#else
            faceCascade.load("/home/fullrange/fri_faceDetector/haarcascades/haarcascade_frontalface_alt.xml") &&
            profileCascade.load("/home/fullrange/fri_faceDetector/haarcascades/haarcascade_profileface_alt.xml") &&
#endif
            eyeCascade.load("/home/fullrange/fri_faceDetector/haarcascades/haarcascade_eye.xml") &&
            eyeGlassCascade.load("/home/fullrange/fri_faceDetector/haarcascades/haarcascade_eye_tree_eyeglasses.xml") &&
            mouthCascade.load("/home/fullrange/fri_faceDetector/haarcascades/haarcascade_mouth.xml") &&
            leftEyeCascade.load("/home/fullrange/fri_faceDetector/haarcascades/haarcascade_lefteye.xml") &&
            rightEyeCascade.load("/home/fullrange/fri_faceDetector/haarcascades/haarcascade_righteye.xml") &&
            noseCascade.load("/home/fullrange/fri_faceDetector/haarcascades/haarcascade_nose.xml");
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

#ifdef TEMPLATE_MATCHING
#ifdef GPU
    static vector<Rect> previousFaces;
    static vector<cuda::GpuMat> faceTemplates;
    static uint frameAt = 1;
#endif
#endif

    vector<Rect> filteredFaces;
    Mat frameGray;

    float scale = 0;

    Face toPush;

    cvtColor(*frame, frameGray, CV_BGR2GRAY);
#ifdef GPU
    cuda::GpuMat frameGpu(frameGray);
#endif

#ifdef EQUALIZE_HIST
#ifdef GPU
    static auto clahe = cuda::createCLAHE(2, Size(8, 8));
    clahe->apply(frameGpu, frameGpu);
#ifdef OUTPUT
    imshow("Debug", frameGpu);
#endif
#else
    static auto clahe = createCLAHE(2, Size(8, 8));
    clahe->apply(frameGray, frameGray);
#ifdef OUTPUT
    imshow("Debug", frameGray);
#endif
//    equalizeHist(frameGray, frameGray);
#endif

#ifdef LOWER_CONTRAST

#ifdef OUTPUT
    imshow("Debug", frameGray);
#endif
#endif

#endif

    // Frontal faces
#ifdef GPU
    static cuda::Stream gpuStream;

    static Ptr<cuda::CascadeClassifier> gpuFaceCascade = cuda::CascadeClassifier::create("/home/fullrange/fri_faceDetector/haarcascadesgpu/haarcascade_frontalface_alt.xml");
    cuda::GpuMat facesGpu;
    gpuFaceCascade->setScaleFactor(1.1);
    gpuFaceCascade->setMinNeighbors(4);
    gpuFaceCascade->setMinObjectSize(Size(50,50));
    gpuFaceCascade->detectMultiScale(frameGpu, facesGpu);
    gpuFaceCascade->convert(facesGpu, faces);

#else
#ifdef LBP

    faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0| CASCADE_SCALE_IMAGE, Size(2, 2));

#else

    faceCascade.detectMultiScale(frameGray, faces, 1.2, 3, 0| CV_HAAR_SCALE_IMAGE, Size(2, 2));

#endif
#endif

#ifdef TEMPLATE_MATCHING
#ifdef GPU
    cuda::GpuMat templateResult;
    for (uint i = 0; i < faceTemplates.size(); i++){
        Ptr<cuda::TemplateMatching> matcher = cuda::createTemplateMatching(CV_8U, CV_TM_CCOEFF);

        Rect imageRect(0, 0, frameGray.cols, frameGray.rows);
        Rect faceRect = previousFaces[i];
        Rect roiRect(faceRect);
        roiRect.x -= TEMPLATE_MATCHING_ROI_MARGIN;
        roiRect.y -= TEMPLATE_MATCHING_ROI_MARGIN;
        roiRect.width += TEMPLATE_MATCHING_ROI_MARGIN * 2;
        roiRect.height += TEMPLATE_MATCHING_ROI_MARGIN * 2;
        roiRect &= imageRect;

        matcher->match(frameGpu(roiRect), faceTemplates[i], templateResult);

        Point minLoc;
        Point maxLoc;
        double minVal;
        double maxVal;

        cuda::minMaxLoc(templateResult, &minVal, &maxVal, &minLoc, &maxLoc);
        if (maxVal < TEMPLATE_MATCHING) {
            continue;
        }

        Rect foundFace(maxLoc.x + roiRect.x, maxLoc.y + roiRect.y, faceRect.width, faceRect.height);

        if (frameAt % TEMPLATE_MATCHING_CLEANUP_INTERVAL == 0) {
            cuda::GpuMat subFacesGpu;
            vector<Rect> subFaces;
            gpuFaceCascade->detectMultiScale(frameGpu(foundFace), subFacesGpu);
            gpuFaceCascade->setScaleFactor(1.1);
            gpuFaceCascade->setMinNeighbors(8);
            gpuFaceCascade->setMinObjectSize(Size(50,50));
            gpuFaceCascade->convert(subFacesGpu, subFaces);
            if (subFaces.size() > 0) {
                faces.push_back(foundFace);
            }
        } else {
            faces.push_back(foundFace);
        }
    }
    frameAt++;
#endif
#endif

#ifdef SIDE_FACES
    // Faces turned to the left
#ifdef GPU
    static Ptr<cuda::CascadeClassifier> gpuProfileFaceCascade = cuda::CascadeClassifier::create("/home/fullrange/fri_faceDetector/haarcascadesgpu/haarcascade_profileface.xml");
    cuda::GpuMat profileFacesGpu;
    gpuProfileFaceCascade->setScaleFactor(1.1);
    gpuProfileFaceCascade->setMinNeighbors(4);
    gpuProfileFaceCascade->setMinObjectSize(Size(5,5));
    gpuProfileFaceCascade->detectMultiScale(frameGpu, profileFacesGpu);
    gpuProfileFaceCascade->convert(profileFacesGpu, profileFaces);
#else
    profileCascade.detectMultiScale(frameGray, profileFaces, 1.3, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));
#endif
    for (uint i = 0; i < profileFaces.size(); i++){
        profileFaces[i].x = profileFaces[i].x - profileFaces[i].width * 1/7;
        profileFaces[i].width *= SIDE_FACE_SCALE_LEFT;
        profileFaces[i].height *= SIDE_FACE_SCALE_LEFT;
        profileFaces[i].y += profileFaces[i].height * (1 - SIDE_FACE_SCALE_LEFT) / 2;
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
    if (faces.size() == 0){
        faces.insert(faces.end(), profileFaces.begin(), profileFaces.end());
    }

    // Faces turned to the right

#ifdef GPU
    cuda::GpuMat profileFaces2Gpu;
    cuda::flip(frameGpu, frameGpu, 1);
    gpuProfileFaceCascade->detectMultiScale(frameGpu, profileFaces2Gpu);
    gpuProfileFaceCascade->convert(profileFaces2Gpu, profileFaces);
#else
    Mat flipped;
    flip(frameGray, flipped, 1);
    profileCascade.detectMultiScale(flipped, profileFaces, 1.3, 3, 0| CV_HAAR_SCALE_IMAGE, Size(10, 10));
#endif

    for (uint i = 0; i < profileFaces.size(); i++){
        profileFaces[i].x = frameGray.cols - profileFaces[i].x - profileFaces[i].width * 2/3;
        profileFaces[i].width *= SIDE_FACE_SCALE_RIGHT;
        profileFaces[i].height *= SIDE_FACE_SCALE_RIGHT;
        profileFaces[i].y += profileFaces[i].height * (1 - SIDE_FACE_SCALE_RIGHT) / 2;
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
    if (faces.size() == 0){
        faces.insert(faces.end(), profileFaces.begin(), profileFaces.end());
    }
#endif


    // Eliminate "double" faces
    vector<bool> overlapTable;

    for (uint i = 0; i < faces.size(); i++) {
        overlapTable.push_back(false);
    }

    for (uint i = 0; i < faces.size(); i++) {
        for (uint j = i+1; j < faces.size(); j++) {
            Rect intersectRect = faces[i] & faces[j];
            Rect unionRect = faces[i] | faces[j];
            double intersectPercent = (double)intersectRect.area() / (double)faces[i].area();
            if (intersectPercent > OVERLAP_TOLERANCE){
                overlapTable[j] = true;
            }
        }
    }
    for (uint i = 0; i < faces.size(); i++) {
        if (overlapTable[i] == false) {
            filteredFaces.push_back(faces[i]);
        }
    }

    toReturn->clear();

    for(uint i = 0; i < filteredFaces.size(); ++i){
        Rect face = filteredFaces[i];

        scale = (float)face.height / (float)frame->rows;

        // Try to detect eyes
        Rect roi(face.x, face.y, face.width, face.height);
        Rect roiRightEye(face.x + face.width * 1/3, face.y, face.width * 2/3, face.height * 2/3);
        Rect roiLeftEye(face.x, face.y, face.width * 2/3, face.height * 2/3);
        Rect roiMouth(face.x, face.y + face.height * 1/2, face.width, face.height * 1/2);

        Rect imageRect(0, 0, frameGray.cols, frameGray.rows);

        vector<Rect>rightEyes;
        vector<Rect>leftEyes;
        vector<Rect>mouths;
//        vector<Rect>noses;

        roi &= imageRect;
        roiRightEye &= imageRect;
        roiLeftEye &= imageRect;
        roiMouth &= imageRect;

#ifdef ROTATION
        rightEyeCascade.detectMultiScale(frameGray(roiRightEye), rightEyes, 1.3, 2, 0 | CV_HAAR_SCALE_IMAGE);
        leftEyeCascade.detectMultiScale(frameGray(roiLeftEye), leftEyes, 1.3, 5, 0 | CV_HAAR_SCALE_IMAGE);
        mouthCascade.detectMultiScale(frameGray(roiMouth), mouths, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(face.width * 0.1, face.width * 0.1),  Size(face.width * 0.8, face.width * 0.8));
#endif

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
        #ifdef OUTPUT
        // imshow("Face", output);
        #endif

        toPush = Face(face.x, face.y, scale, finalAngle);
        toReturn->push_back(toPush);

    }

#ifdef TEMPLATE_MATCHING
#ifdef GPU
    if (filteredFaces.size() > 0) {
        previousFaces.clear();
        faceTemplates.clear();

        previousFaces.insert(previousFaces.end(), filteredFaces.begin(), filteredFaces.end());

        for (uint i = 0; i < previousFaces.size(); i++){
            Rect faceRect = previousFaces[i];
            cuda::GpuMat foundFace;
            frameGpu(faceRect).copyTo(foundFace);
            faceTemplates.push_back(foundFace);
        }
    }
#endif
#endif
}
