#ifndef DETECTIONSTRUCT_H
#define DETECTIONSTRUCT_H

#include <QObject>
#include <QString>

// =========================================================
// Struct for efficient data passing to QML
// =========================================================
struct Detection {
    Q_GADGET
    Q_PROPERTY(int classId MEMBER classId)
    Q_PROPERTY(float confidence MEMBER confidence)
    Q_PROPERTY(QString label MEMBER label)
    Q_PROPERTY(float x MEMBER x)
    Q_PROPERTY(float y MEMBER y)
    Q_PROPERTY(float w MEMBER w)
    Q_PROPERTY(float h MEMBER h)

public:
    int classId;
    float confidence;
    QString label;
    float x;
    float y;
    float w;
    float h;
};
Q_DECLARE_METATYPE(Detection)

#endif // DETECTIONSTRUCT_H
