#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQuickStyle>

using namespace Qt::StringLiterals;

int main(int argc, char *argv[]) {
  QGuiApplication app(argc, argv);
  
  QQuickStyle::setStyle("Basic");

  QQmlApplicationEngine engine;

  // Load the QML file from the module path defined in CMake
  // Note the added "/qt/qml/" at the start
  const QUrl url(u"qrc:/qt/qml/CameraModule/content/Main.qml"_s);

  QObject::connect(
      &engine, &QQmlApplicationEngine::objectCreated, &app,
      [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
          QCoreApplication::exit(-1);
      },
      Qt::QueuedConnection);

  engine.load(url);

  return app.exec();
}
