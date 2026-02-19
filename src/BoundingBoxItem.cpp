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

QVariantList BoundingBoxItem::detections() const
{
    return m_detections;
}

void BoundingBoxItem::setDetections(const QVariantList &detections)
{
    if (m_detections != detections) {
        m_detections = detections;
        emit detectionsChanged();
        update(); // Schedule a scene graph update
    }
}

QSGNode *BoundingBoxItem::updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *)
{
    // Define a custom node structure to hold all our boxes
    // Root -> [Box1, Box2, Box3...]
    // Better: Single GeometryNode with GL_LINES or GL_TRIANGLES for all boxes?
    // Batching is better. Let's use ONE node with lines for the rectangles.
    
    QSGGeometryNode *node = static_cast<QSGGeometryNode *>(oldNode);
    int detectionCount = m_detections.size();

    if (detectionCount == 0) {
        if (node) delete node;
        return nullptr;
    }

    if (!node) {
        node = new QSGGeometryNode;
        // 4 lines per box * 2 vertices per line = 8 vertices per detection
        // Use defaultAttributes_ColoredPoint2D() for modern Qt
        QSGGeometry *geometry = new QSGGeometry(QSGGeometry::defaultAttributes_ColoredPoint2D(), 0);
        geometry->setLineWidth(3);
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
    for (const QVariant &var : m_detections) {
        if (var.canConvert<Detection>()) {
            Detection det = var.value<Detection>();

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
    }
    
    // Safety check just in case allocations don't match loop
    // geometry->markVertexDataDirty(); // Included in allocate() usually, but safe to call if modifying existing
    node->markDirty(QSGNode::DirtyGeometry);

    return node;
}
