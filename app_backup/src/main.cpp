#ifdef _WIN32
#include <windows.h>
#endif

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQuickStyle>
#include <string>

using namespace Qt::StringLiterals;

int main(int argc, char *argv[]) {
#ifdef _WIN32
  // Configure DLL search path for organized libs/ structure
  wchar_t path[MAX_PATH];
  if (GetModuleFileNameW(NULL, path, MAX_PATH) != 0) {
      std::wstring binDir = path;
      size_t lastBackslash = binDir.find_last_of(L"\\/");
      if (lastBackslash != std::wstring::npos) {
          binDir = binDir.substr(0, lastBackslash);
      }
      // Point secondary DLL search to libs/ for dynamically loaded plugins
      std::wstring libsDir = binDir + L"\\libs";
      SetDllDirectoryW(libsDir.c_str());
  }
#endif
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
