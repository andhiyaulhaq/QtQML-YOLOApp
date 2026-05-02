#pragma once

#include <QSize>

struct CameraConfig {
    int deviceId = 0;
    QSize resolution = QSize(640, 480);
    double fps = 30.0;
};
