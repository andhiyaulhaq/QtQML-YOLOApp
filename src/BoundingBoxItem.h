#ifndef BOUNDINGBOXITEM_H
#define BOUNDINGBOXITEM_H

#include <QQuickItem>
#include <QSGNode>
#include <QSGFlatColorMaterial>
#include <QSGGeometryNode>
#include <QSGGeometry>
#include "VideoController.h"
#include "DetectionListModel.h"

class BoundingBoxItem : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(QObject* detections READ detections WRITE setDetections NOTIFY detectionsChanged)
    QML_ELEMENT

public:
    BoundingBoxItem(QQuickItem *parent = nullptr);

    QObject* detections() const;
    void setDetections(QObject *detections);

    QSGNode *updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) override;

signals:
    void detectionsChanged();

private slots:
    void onModelUpdated();

private:
    DetectionListModel* m_model = nullptr;
};

#endif // BOUNDINGBOXITEM_H
