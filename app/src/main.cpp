#ifdef _WIN32
#include <windows.h>
#endif

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QtQml>
#include <QQuickStyle>
#include "shared/application/AppController.h"
#include "features/camera/application/YoloCameraController.h"
#include "features/monitoring/application/MonitoringController.h"
#include "features/detection/application/DetectionController.h"
#include "features/detection/ui/DetectionListModel.h"
#include "features/detection/ui/DetectionOverlayItem.h"
#include "features/detection/domain/DetectionResult.h"
#include "features/detection/domain/InferenceTiming.h"
#include "features/detection/domain/InferenceConfig.h"

#include "features/detection/domain/TaskType.h"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    QQuickStyle::setStyle("Basic");
    qRegisterMetaType<Detection>("Detection");
    qRegisterMetaType<YoloTask::TaskType>("YoloTask::TaskType");
    qRegisterMetaType<YoloTask::RuntimeType>("YoloTask::RuntimeType");
    qRegisterMetaType<std::vector<DetectionResult>>("std::vector<DetectionResult>");
    qRegisterMetaType<InferenceTiming>("InferenceTiming");
    qRegisterMetaType<InferenceConfig>("InferenceConfig");
    qRegisterMetaType<std::shared_ptr<cv::Mat>>("std::shared_ptr<cv::Mat>");
    qRegisterMetaType<std::shared_ptr<std::vector<DetectionResult>>>("std::shared_ptr<std::vector<DetectionResult>>");

    qDebug() << "Registering QML types...";
    qmlRegisterType<YoloCameraController>("CameraModule", 1, 0, "YoloCameraController");
    qmlRegisterType<MonitoringController>("CameraModule", 1, 0, "MonitoringController");
    qmlRegisterType<DetectionController>("CameraModule", 1, 0, "DetectionController");
    qmlRegisterType<DetectionListModel>("CameraModule", 1, 0, "DetectionListModel");
    qmlRegisterType<DetectionOverlayItem>("CameraModule", 1, 0, "DetectionOverlayItem");
    qmlRegisterUncreatableMetaObject(YoloTask::staticMetaObject, "CameraModule", 1, 0, "YoloTask", "Access to enums");

    QQmlApplicationEngine engine;
    
    AppController controller(&engine);
    controller.initialize();

    const QUrl url(u"qrc:/qt/qml/CameraModule/src/ui/Main.qml"_qs);
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
        &app, [url](QObject *obj, const QUrl &objUrl) {
            if (!obj && url == objUrl)
                QCoreApplication::exit(-1);
        }, Qt::QueuedConnection);
    qDebug() << "Main: Loading QML from" << url;
    engine.load(url);
    
    if (engine.rootObjects().isEmpty()) {
        qDebug() << "Main: Failed to load QML!";
        return -1;
    }

    qDebug() << "Main: QML loaded, entering event loop.";
    return app.exec();
}
