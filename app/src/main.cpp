#ifdef _WIN32
#include <windows.h>
#endif

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QtQml>
#include <QQuickStyle>
#include "shared/application/AppController.h"
#include "features/capture/application/YoloCameraController.h"
#include "features/monitoring/application/MonitoringController.h"
#include "features/inference/application/InferenceController.h"
#include "features/inference/ui/InferenceListModel.h"
#include "features/inference/ui/InferenceOverlayItem.h"
#include "features/inference/domain/InferenceResult.h"
#include "features/inference/domain/InferenceTiming.h"
#include "features/inference/domain/InferenceConfig.h"
#include "features/inference/domain/VisionObject.h"
#include "features/inference/domain/TaskType.h"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    QQuickStyle::setStyle("Basic");
    qRegisterMetaType<VisionObject>("VisionObject");
    qRegisterMetaType<YoloTask::TaskType>("YoloTask::TaskType");
    qRegisterMetaType<YoloTask::RuntimeType>("YoloTask::RuntimeType");
    qRegisterMetaType<std::vector<InferenceResult>>("std::vector<InferenceResult>");
    qRegisterMetaType<InferenceTiming>("InferenceTiming");
    qRegisterMetaType<InferenceConfig>("InferenceConfig");
    qRegisterMetaType<std::shared_ptr<cv::Mat>>("std::shared_ptr<cv::Mat>");
    qRegisterMetaType<std::shared_ptr<std::vector<InferenceResult>>>("std::shared_ptr<std::vector<InferenceResult>>");

    qDebug() << "Registering QML types...";
    qmlRegisterType<YoloCameraController>("CameraModule", 1, 0, "YoloCameraController");
    qmlRegisterType<MonitoringController>("CameraModule", 1, 0, "MonitoringController");
    qmlRegisterType<InferenceController>("CameraModule", 1, 0, "InferenceController");
    qmlRegisterType<InferenceListModel>("CameraModule", 1, 0, "InferenceListModel");
    qmlRegisterType<InferenceOverlayItem>("CameraModule", 1, 0, "InferenceOverlayItem");
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
