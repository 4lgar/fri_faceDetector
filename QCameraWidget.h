#ifndef CAMERA_H_
#define CAMERA_H_

#include <QWidget>
#include <QVBoxLayout>
#include "QCameraWindow.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "Face.h"


class QCameraWidget : public QWidget
{
    Q_OBJECT
    private:
        QCameraWindow *cvwidget;
        CvCapture *camera;
        QList<Face *> *faces;
        
    public:
        QCameraWidget(CvCapture *cam, QList<Face *> *fa, QWidget *parent=0);
         
    protected:
        void timerEvent(QTimerEvent*);        
};


#endif /*CAMERA_H_*/
