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
    // Each box is 4 lines: (TL-TR), (TR-BR), (BR-BL), (BL-TL)
    // 8 vertices per box.
    geometry->allocate(detectionCount * 8);

    QSGGeometry::ColoredPoint2D *vertices = geometry->vertexDataAsColoredPoint2D();

    int idx = 0;
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
        vertices[idx].set(x, y, r, g, b, a); idx++;
        vertices[idx].set(x + w, y, r, g, b, a); idx++;

        // Right Line
        vertices[idx].set(x + w, y, r, g, b, a); idx++;
        vertices[idx].set(x + w, y + h, r, g, b, a); idx++;

        // Bottom Line
        vertices[idx].set(x + w, y + h, r, g, b, a); idx++;
        vertices[idx].set(x, y + h, r, g, b, a); idx++;

        // Left Line
        vertices[idx].set(x, y + h, r, g, b, a); idx++;
        vertices[idx].set(x, y, r, g, b, a); idx++;
    }
    
    // Safety check just in case allocations don't match loop
    node->markDirty(QSGNode::DirtyGeometry);

    return node;
}
