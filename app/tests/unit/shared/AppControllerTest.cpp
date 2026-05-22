#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "AppController.h"
#include <QQmlApplicationEngine>
#include <QCoreApplication>

TEST(AppControllerTest, CleanLifecycle) {
    // AppController needs a QCoreApplication or QGuiApplication for signals/threads
    int argc = 1;
    char* argv[] = {(char*)"test"};
    QCoreApplication app(argc, argv);

    QQmlApplicationEngine engine;
    
    // We scope the controller so it gets destroyed
    {
        AppController controller(&engine);
        controller.initialize();
        
        // Let threads start
        QCoreApplication::processEvents();
    }
    
    // If it didn't hang or crash, it's successful.
    // The destructor should stop and join all internal threads.
    SUCCEED();
}
