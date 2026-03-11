#include "BoundingBoxItem.h"
#include <QFont>
#include <QFontMetrics>
#include <QSGVertexColorMaterial> // Added missing header
#include <QSGGeometryNode>
#include <QSGGeometry>

BoundingBoxItem::BoundingBoxItem(QQuickItem *parent)
    : QQuickItem(parent)
{
    // Important: We need this flag to participate in the scenegraph update loop
    setFlag(ItemHasContents, true);
}

QObject* BoundingBoxItem::detections() const
{
    return m_model;
}

void BoundingBoxItem::setDetections(QObject *detections)
{
    DetectionListModel* newModel = qobject_cast<DetectionListModel*>(detections);
    // qDebug() << "BoundingBoxItem::setDetections called with ptr:" << detections << " casted:" << newModel;
    
    if (m_model != newModel) {
        if (m_model) {
            disconnect(m_model, nullptr, this, nullptr);
        }
        m_model = newModel;
        if (m_model) {
            connect(m_model, &QAbstractListModel::modelReset, this, &BoundingBoxItem::onModelUpdated);
            connect(m_model, &QAbstractListModel::layoutChanged, this, &BoundingBoxItem::onModelUpdated);
            connect(m_model, &QAbstractListModel::rowsInserted, this, &BoundingBoxItem::onModelUpdated);
            connect(m_model, &QAbstractListModel::rowsRemoved, this, &BoundingBoxItem::onModelUpdated);
            connect(m_model, &QAbstractListModel::dataChanged, this, &BoundingBoxItem::onModelUpdated);
            // qDebug() << "BoundingBoxItem: Model connected";
        }
        emit detectionsChanged();
        update(); // Schedule a scene graph update
    }
}

void BoundingBoxItem::onModelUpdated()
{
    // qDebug() << "BoundingBoxItem::onModelUpdated triggered"; // Noisy if frequent
    update();
}

QSGNode *BoundingBoxItem::updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *)
{
    QSGGeometryNode *node = static_cast<QSGGeometryNode *>(oldNode);
    
    // Safety check
    if (!m_model) {
        if (node) delete node;
        return nullptr;
    }

    const auto& detections = m_model->getDetections();
    int detectionCount = detections.size();
    
    // qDebug() << "BoundingBoxItem::updatePaintNode: detections=" << detectionCount;

    if (detectionCount == 0) {
        if (node) {
            node->geometry()->allocate(0);
            node->markDirty(QSGNode::DirtyGeometry);
        }
        return node;
    }

    if (!node) {
        node = new QSGGeometryNode;
        // 4 lines per box * 2 vertices per line = 8 vertices per detection
        // Use defaultAttributes_ColoredPoint2D() for modern Qt
        QSGGeometry *geometry = new QSGGeometry(QSGGeometry::defaultAttributes_ColoredPoint2D(), 0);
        geometry->setLineWidth(1);
        geometry->setDrawingMode(QSGGeometry::DrawLines); // Use Qt enum instead of GL_LINES
        node->setGeometry(geometry);
        node->setFlag(QSGNode::OwnsGeometry);

        QSGVertexColorMaterial *material = new QSGVertexColorMaterial;
        node->setMaterial(material);
        node->setFlag(QSGNode::OwnsMaterial);
    }

    QSGGeometry *geometry = node->geometry();

    // Calculate total vertices needed
    int totalVertices = 0;
    for (const auto &det : detections) {
        totalVertices += 8; // 4 lines for bounding box
        if (det.keyPoints.size() == 17) {
            totalVertices += 19 * 2; // 19 skeleton edges
            totalVertices += 17 * 4; // 17 crosses (4 vertices per cross)
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

    for (const auto &det : detections) {
        // Convert normalized to local coords
        float x = det.x * width();
        float y = det.y * height();
        float w = det.w * width();
        float h = det.h * height();

        // Color generation
        int hue = (det.classId * 60) % 360;
        QColor color = QColor::fromHsl(hue, 255, 127);
        quint8 r = color.red();
        quint8 g = color.green();
        quint8 b = color.blue();
        quint8 a = 255;

        // Top Line
        vertices[idx++].set(x, y, r, g, b, a);
        vertices[idx++].set(x + w, y, r, g, b, a);

        // Right Line
        vertices[idx++].set(x + w, y, r, g, b, a);
        vertices[idx++].set(x + w, y + h, r, g, b, a);

        // Bottom Line
        vertices[idx++].set(x + w, y + h, r, g, b, a);
        vertices[idx++].set(x, y + h, r, g, b, a);

        // Left Line
        vertices[idx++].set(x, y + h, r, g, b, a);
        vertices[idx++].set(x, y, r, g, b, a);

        // Pose Skeletons and KeyPoints
        if (det.keyPoints.size() == 17) {
            std::vector<QPointF> unnormalizedKpts;
            for (int k = 0; k < 17; k++) {
                unnormalizedKpts.push_back(QPointF(
                    det.keyPoints[k].x() * width(), 
                    det.keyPoints[k].y() * height()
                ));
            }

            // Draw skeleton edges
            for (int e = 0; e < 19; e++) {
                int p1 = skeleton[e][0];
                int p2 = skeleton[e][1];
                
                // If point exists (> 0,0 normalized)
                if (det.keyPoints[p1].x() > 0.001f || det.keyPoints[p1].y() > 0.001f) {
                    if (det.keyPoints[p2].x() > 0.001f || det.keyPoints[p2].y() > 0.001f) {
                        vertices[idx++].set(unnormalizedKpts[p1].x(), unnormalizedKpts[p1].y(), 0, 255, 0, 255);
                        vertices[idx++].set(unnormalizedKpts[p2].x(), unnormalizedKpts[p2].y(), 0, 255, 0, 255);
                        continue;
                    }
                }
                // Skip invalid lines by rendering zero-length
                vertices[idx++].set(0, 0, 0, 0, 0, 0);
                vertices[idx++].set(0, 0, 0, 0, 0, 0);
            }

            // Draw points as small crosses
            float size = 5.0f;
            for (int k = 0; k < 17; k++) {
                if (det.keyPoints[k].x() > 0.001f || det.keyPoints[k].y() > 0.001f) {
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
