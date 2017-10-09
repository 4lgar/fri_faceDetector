
#ifndef QCAMERAWINDOW_H
#define QCAMERAWINDOW_H

#include <opencv/cv.h>
#include <QPixmap>
#include <QLabel>
#include <QWidget>
#include <QVBoxLayout>
#include <QImage>

class QCameraWindow : public QWidget {
    private:
        QLabel *imagelabel;
        QVBoxLayout *layout;
        
        QImage image;
        
    public:
        QCameraWindow(QWidget *parent = 0);
        ~QCameraWindow(void);
        void putImage(IplImage *);
}; 

#endif
