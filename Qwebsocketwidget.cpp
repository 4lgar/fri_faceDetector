#include "qwebsocketwidget.h"

QWebsocketWidget::QWebsocketWidget()
{
    camera = cam;
    QVBoxLayout *layout = new QVBoxLayout;
    cvwidget = new QCameraWindow(this);
    layout->addWidget(cvwidget);
    setLayout(layout);
    resize(500, 400);

    startTimer(10);  // 0.1-second timer
}
