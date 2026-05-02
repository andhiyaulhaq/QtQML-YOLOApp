#pragma once

#include <QObject>
#include <QString>
#include <QPointF>
#include <QList>

struct Detection {
    Q_GADGET
    Q_PROPERTY(int classId READ classId CONSTANT)
    Q_PROPERTY(float confidence READ confidence CONSTANT)
    Q_PROPERTY(QString label READ label CONSTANT)
    Q_PROPERTY(float x READ x CONSTANT)
    Q_PROPERTY(float y READ y CONSTANT)
    Q_PROPERTY(float w READ w CONSTANT)
    Q_PROPERTY(float h READ h CONSTANT)
    Q_PROPERTY(QList<QPointF> keyPoints READ keyPoints CONSTANT)

public:
    int classId() const { return m_classId; }
    float confidence() const { return m_confidence; }
    QString label() const { return m_label; }
    float x() const { return m_x; }
    float y() const { return m_y; }
    float w() const { return m_w; }
    float h() const { return m_h; }
    QList<QPointF> keyPoints() const { return m_keyPoints; }

    int m_classId;
    float m_confidence;
    QString m_label;
    float m_x, m_y, m_w, m_h;
    QList<QPointF> m_keyPoints;
};

Q_DECLARE_METATYPE(Detection)
