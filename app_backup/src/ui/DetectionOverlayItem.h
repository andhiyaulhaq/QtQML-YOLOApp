#ifndef DETECTIONOVERLAYITEM_H
#define DETECTIONOVERLAYITEM_H

#include <QQuickItem>
#include <QSGNode>
#include <QSGFlatColorMaterial>
#include <QSGGeometryNode>
#include <QSGGeometry>
#include "VideoController.h"
#include "DetectionListModel.h"

class DetectionOverlayItem : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(QObject* detections READ detections WRITE setDetections NOTIFY detectionsChanged)
    QML_ELEMENT

public:
    DetectionOverlayItem(QQuickItem *parent = nullptr);

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

#endif // DETECTIONOVERLAYITEM_H
