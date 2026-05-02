#pragma once

#include <QQuickItem>
#include <QSGNode>
#include <QSGFlatColorMaterial>
#include <QSGGeometryNode>
#include <QSGGeometry>
#include "../ui/DetectionListModel.h"

class DetectionOverlayItem : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(QObject* detections READ detections WRITE setDetections NOTIFY detectionsChanged)

public:
    explicit DetectionOverlayItem(QQuickItem *parent = nullptr);

    QObject* detections() const { return m_model; }
    void setDetections(QObject *detections);

    QSGNode *updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) override;

signals:
    void detectionsChanged();

private slots:
    void onModelUpdated();

private:
    DetectionListModel* m_model = nullptr;
};
