#include "DetectionOverlayItem.h"
#include <QSGVertexColorMaterial>
#include <QSGGeometryNode>
#include <QSGGeometry>
#include <QColor>

DetectionOverlayItem::DetectionOverlayItem(QQuickItem *parent)
    : QQuickItem(parent)
{
    setFlag(ItemHasContents, true);
}

void DetectionOverlayItem::setDetections(QObject *detections)
{
    DetectionListModel* newModel = qobject_cast<DetectionListModel*>(detections);
    
    if (m_model != newModel) {
        if (m_model) {
            disconnect(m_model, nullptr, this, nullptr);
        }
        m_model = newModel;
        if (m_model) {
            connect(m_model, &QAbstractListModel::modelReset, this, &DetectionOverlayItem::onModelUpdated);
            connect(m_model, &QAbstractListModel::layoutChanged, this, &DetectionOverlayItem::onModelUpdated);
            connect(m_model, &QAbstractListModel::rowsInserted, this, &DetectionOverlayItem::onModelUpdated);
            connect(m_model, &QAbstractListModel::rowsRemoved, this, &DetectionOverlayItem::onModelUpdated);
            connect(m_model, &QAbstractListModel::dataChanged, this, &DetectionOverlayItem::onModelUpdated);
        }
        emit detectionsChanged();
        update(); 
    }
}

void DetectionOverlayItem::onModelUpdated()
{
    update();
}

QSGNode *DetectionOverlayItem::updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *)
{
    QSGGeometryNode *node = static_cast<QSGGeometryNode *>(oldNode);
    
    if (!m_model) {
        if (node) delete node;
        return nullptr;
    }

    const auto& detections = m_model->getDetections();
    int detectionCount = static_cast<int>(detections.size());
    
    if (detectionCount == 0) {
        if (node) {
            node->geometry()->allocate(0);
            node->markDirty(QSGNode::DirtyGeometry);
        }
        return node;
    }

    if (!node) {
        node = new QSGGeometryNode;
        QSGGeometry *geometry = new QSGGeometry(QSGGeometry::defaultAttributes_ColoredPoint2D(), 0);
        geometry->setLineWidth(2);
        geometry->setDrawingMode(QSGGeometry::DrawLines);
        node->setGeometry(geometry);
        node->setFlag(QSGNode::OwnsGeometry);

        QSGVertexColorMaterial *material = new QSGVertexColorMaterial;
        node->setMaterial(material);
        node->setFlag(QSGNode::OwnsMaterial);
    }

    QSGGeometry *geometry = node->geometry();

    int totalVertices = 0;
    for (const auto &det : detections) {
        totalVertices += 8; 
        if (det.keyPoints().size() == 17) {
            totalVertices += 19 * 2; 
            totalVertices += 17 * 4; 
        }
    }
    
    geometry->allocate(totalVertices);
    QSGGeometry::ColoredPoint2D *vertices = geometry->vertexDataAsColoredPoint2D();

    int idx = 0;
    const int skeleton[][2] = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, 
        {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, 
        {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}
    };

    float itemW = width();
    float itemH = height();
    float renderW = itemW;
    float renderH = itemH;
    float offsetX = 0;
    float offsetY = 0;

    QSize frameSize = m_model->frameSize();
    if (!frameSize.isEmpty() && itemW > 0 && itemH > 0) {
        float videoAspectRatio = static_cast<float>(frameSize.width()) / static_cast<float>(frameSize.height());
        float itemAspectRatio = itemW / itemH;

        if (videoAspectRatio > itemAspectRatio) {
            renderH = itemW / videoAspectRatio;
            offsetY = (itemH - renderH) / 2.0f;
        } else {
            renderW = itemH * videoAspectRatio;
            offsetX = (itemW - renderW) / 2.0f;
        }
    }

    for (const auto &det : detections) {
        float x = offsetX + det.x() * renderW;
        float y = offsetY + det.y() * renderH;
        float w = det.w() * renderW;
        float h = det.h() * renderH;

        int hue = (det.classId() * 60) % 360;
        QColor color = QColor::fromHsl(hue, 255, 127);
        quint8 r = color.red();
        quint8 g = color.green();
        quint8 b = color.blue();
        quint8 a = 255;

        // Top
        vertices[idx++].set(x, y, r, g, b, a);
        vertices[idx++].set(x + w, y, r, g, b, a);
        // Right
        vertices[idx++].set(x + w, y, r, g, b, a);
        vertices[idx++].set(x + w, y + h, r, g, b, a);
        // Bottom
        vertices[idx++].set(x + w, y + h, r, g, b, a);
        vertices[idx++].set(x, y + h, r, g, b, a);
        // Left
        vertices[idx++].set(x, y + h, r, g, b, a);
        vertices[idx++].set(x, y, r, g, b, a);

        if (det.keyPoints().size() == 17) {
            std::vector<QPointF> unnormalizedKpts;
            const auto& kpts = det.keyPoints();
            for (int k = 0; k < 17; k++) {
                unnormalizedKpts.push_back(QPointF(
                    offsetX + kpts[k].x() * renderW, 
                    offsetY + kpts[k].y() * renderH
                ));
            }

            for (int e = 0; e < 19; e++) {
                int p1 = skeleton[e][0];
                int p2 = skeleton[e][1];
                if (kpts[p1].x() > 0.001f || kpts[p1].y() > 0.001f) {
                    if (kpts[p2].x() > 0.001f || kpts[p2].y() > 0.001f) {
                        vertices[idx++].set(unnormalizedKpts[p1].x(), unnormalizedKpts[p1].y(), 0, 255, 0, 255);
                        vertices[idx++].set(unnormalizedKpts[p2].x(), unnormalizedKpts[p2].y(), 0, 255, 0, 255);
                        continue;
                    }
                }
                vertices[idx++].set(0, 0, 0, 0, 0, 0);
                vertices[idx++].set(0, 0, 0, 0, 0, 0);
            }

            float size = 4.0f;
            for (int k = 0; k < 17; k++) {
                if (kpts[k].x() > 0.001f || kpts[k].y() > 0.001f) {
                    float kx = unnormalizedKpts[k].x();
                    float ky = unnormalizedKpts[k].y();
                    vertices[idx++].set(kx - size, ky, 255, 0, 0, 255);
                    vertices[idx++].set(kx + size, ky, 255, 0, 0, 255);
                    vertices[idx++].set(kx, ky - size, 255, 0, 0, 255);
                    vertices[idx++].set(kx, ky + size, 255, 0, 0, 255);
                } else {
                    vertices[idx++].set(0, 0, 0, 0, 0, 0);
                    vertices[idx++].set(0, 0, 0, 0, 0, 0);
                    vertices[idx++].set(0, 0, 0, 0, 0, 0);
                    vertices[idx++].set(0, 0, 0, 0, 0, 0);
                }
            }
        }
    }
    
    node->markDirty(QSGNode::DirtyGeometry);
    return node;
}
