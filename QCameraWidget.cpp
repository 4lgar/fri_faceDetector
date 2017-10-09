#include "QCameraWidget.h"

QCameraWidget::QCameraWidget(CvCapture *cam, QList<Face *> *fa, QWidget *parent) : QWidget(parent) {
    faces = fa;

    camera = cam;
    QVBoxLayout *layout = new QVBoxLayout;
    cvwidget = new QCameraWindow(this);
    layout->addWidget(cvwidget);
    setLayout(layout);
    resize(500, 400);

    startTimer(10);  // 0.1-second timer
 }

void QCameraWidget::timerEvent(QTimerEvent*) {
    IplImage *image=cvQueryFrame(camera);
    faces->at(0)->x = (faces->at(0)->x + 1) % 4096;
    faces->at(0)->y = (faces->at(0)->y + 1) % 4096;
    faces->at(0)->rotation = (faces->at(0)->rotation + 1) % 360;
    cvwidget->putImage(image);
}

