#include <QtQuickTest/quicktest.h>
#include <QQmlEngine>
#include "InferenceOverlayItem.h"
#include "InferenceListModel.h"

// Setup hook to register C++ types before tests run
class Setup : public QObject {
public:
    Setup() {
        qmlRegisterType<InferenceOverlayItem>("CameraModule", 1, 0, "InferenceOverlayItem");
        qmlRegisterType<InferenceListModel>("CameraModule", 1, 0, "InferenceListModel");
    }
};

static Setup setup;

QUICK_TEST_MAIN(QMLTests)