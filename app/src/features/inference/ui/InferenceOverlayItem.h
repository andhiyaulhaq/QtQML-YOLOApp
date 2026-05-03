#pragma once

#include <QQuickItem>
#include <QSGNode>
#include <QSGFlatColorMaterial>
#include <QSGGeometryNode>
#include <QSGGeometry>
#include "../ui/InferenceListModel.h"

class InferenceOverlayItem : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(QObject* visionObjects READ visionObjects WRITE setVisionObjects NOTIFY visionObjectsChanged)

public:
    explicit InferenceOverlayItem(QQuickItem *parent = nullptr);

    QObject* visionObjects() const { return m_model; }
    void setVisionObjects(QObject *visionObjects);

    QSGNode *updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) override;

signals:
    void visionObjectsChanged();

private slots:
    void onModelUpdated();

private:
    InferenceListModel* m_model = nullptr;
};
