#pragma once

#include <QObject>

namespace YoloTask {
    Q_NAMESPACE

    enum class TaskType {
        ObjectDetection = 1,
        PoseEstimation = 2,
        ImageSegmentation = 3
    };
    Q_ENUM_NS(TaskType)

    enum class RuntimeType {
        OpenVINO = 0,
        ONNXRuntime = 1
    };
    Q_ENUM_NS(RuntimeType)
}
