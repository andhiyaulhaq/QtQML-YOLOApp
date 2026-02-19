#ifndef BOUNDINGBOXITEM_H
#define BOUNDINGBOXITEM_H

#include <QQuickItem>
#include <QSGNode>
#include <QSGFlatColorMaterial>
#include <QSGGeometryNode>
#include <QSGGeometry>
#include "VideoController.h"

class BoundingBoxItem : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(QVariantList detections READ detections WRITE setDetections NOTIFY detectionsChanged)
    QML_ELEMENT

public:
    BoundingBoxItem(QQuickItem *parent = nullptr);

    QVariantList detections() const;
    void setDetections(const QVariantList &detections);

    QSGNode *updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) override;

signals:
    void detectionsChanged();

private:
    QVariantList m_detections;
};

#endif // BOUNDINGBOXITEM_H
