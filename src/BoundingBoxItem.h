#ifndef BOUNDINGBOXITEM_H
#define BOUNDINGBOXITEM_H

#include <QQuickPaintedItem>
#include <QPainter>
#include "VideoController.h"

class BoundingBoxItem : public QQuickPaintedItem
{
    Q_OBJECT
    Q_PROPERTY(QVariantList detections READ detections WRITE setDetections NOTIFY detectionsChanged)
    QML_ELEMENT

public:
    BoundingBoxItem(QQuickItem *parent = nullptr);

    QVariantList detections() const;
    void setDetections(const QVariantList &detections);

    void paint(QPainter *painter) override;

signals:
    void detectionsChanged();

private:
    QVariantList m_detections;
};

#endif // BOUNDINGBOXITEM_H
