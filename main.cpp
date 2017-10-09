#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <assert.h>
#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include "QCameraWidget.h"
#include "Websocketserver.h"

static int webSocketPort = 1234;

int main(int argc, char **argv) {
    CvCapture *camera = cvCreateCameraCapture(0);
    assert(camera);
    IplImage *image = cvQueryFrame(camera);
    assert(image);

    QList<Face *> *faces = new QList<Face *>();
    faces->append(new Face(1, 2, 30));

    QApplication app(argc, argv);

    WebSocketServer *server = new WebSocketServer(webSocketPort, true, faces);
    QObject::connect(server, &WebSocketServer::closed, &app, &QCoreApplication::quit);

    QCameraWidget *mainWin = new QCameraWidget(camera, faces);
    mainWin->setWindowTitle("FRI | FaceDetector");
    mainWin->show();    

    int retval = app.exec();

    cvReleaseCapture(&camera);

    return retval;
}

