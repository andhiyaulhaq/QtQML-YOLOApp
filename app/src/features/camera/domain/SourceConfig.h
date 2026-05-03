#pragma once

#include <QSize>
#include <QString>

enum class InputSourceType {
    LiveCamera,
    VideoFile,
    ImageFile
};

struct SourceConfig {
    InputSourceType sourceType = InputSourceType::LiveCamera;
    int             deviceId   = 0;
    QString         filePath   = "";       // used when sourceType == VideoFile
    QSize           resolution = QSize(640, 480);
    double          fps        = 30.0;
    bool            loop       = true;     // replay video when EOF reached
};
