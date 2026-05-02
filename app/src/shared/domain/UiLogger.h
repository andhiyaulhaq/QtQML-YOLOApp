#pragma once
#include <QDebug>
#include <QDateTime>

namespace UiLogger {
    inline void log(const char* tier, const QString& message) {
        QString timestamp = QDateTime::currentDateTime().toString("HH:mm:ss.zzz");
        qDebug().noquote() << QString("[%1] [%2] %3").arg(timestamp, tier, message);
    }

    inline void ui(const QString& msg)   { log("UI  ", msg); }
    inline void ctrl(const QString& msg) { log("CTRL", msg); }
}
