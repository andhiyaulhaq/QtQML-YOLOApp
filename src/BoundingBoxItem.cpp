#include "BoundingBoxItem.h"

BoundingBoxItem::BoundingBoxItem(QQuickItem *parent)
    : QQuickPaintedItem(parent)
{
    // Important for performance: only repaint when explicitly updated
    setRenderTarget(QQuickPaintedItem::Image); 
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
        update(); // Trigger repaint
    }
}

void BoundingBoxItem::paint(QPainter *painter)
{
    painter->setRenderHint(QPainter::Antialiasing);

    for (const QVariant &var : m_detections) {
        if (var.canConvert<Detection>()) {
            Detection det = var.value<Detection>();
            
            // Map relative coordinates to item size
            qreal x = det.x * width();
            qreal y = det.y * height();
            qreal w = det.w * width();
            qreal h = det.h * height();

            // Unique color based on class ID
            int hue = (det.classId * 60) % 360;
            QColor color = QColor::fromHsl(hue, 255, 127);
            
            // Draw Box
            QPen pen(color);
            pen.setWidth(3);
            painter->setPen(pen);
            painter->setBrush(Qt::NoBrush);
            painter->drawRect(QRectF(x, y, w, h));

            // Draw Label Background
            QString labelText = QString("%1 %2%").arg(det.label).arg(int(det.confidence * 100));
            QFontMetrics fm(painter->font());
            int textWidth = fm.horizontalAdvance(labelText);
            int textHeight = fm.height();
            
            QRectF labelRect(x, y - textHeight - 4, textWidth + 10, textHeight + 4);
            if (labelRect.top() < 0) labelRect.moveTop(y); // Flip if at top edge

            painter->setBrush(color);
            painter->setPen(Qt::NoPen);
            painter->drawRect(labelRect);

            // Draw Label Text
            painter->setPen(Qt::black);
            painter->drawText(labelRect, Qt::AlignCenter, labelText);
        }
    }
}
