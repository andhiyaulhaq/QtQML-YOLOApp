#include "DetectionListModel.h"
#include <QVariant>
#include <QDebug>
#include <QVideoFrame>
#include <QVideoFrameFormat>
#include "VideoController.h" // For AppConfig

DetectionListModel::DetectionListModel(QObject *parent)
    : QAbstractListModel(parent)
{
}

int DetectionListModel::rowCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;
    return static_cast<int>(m_detections.size());
}

QVariant DetectionListModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid() || index.row() >= static_cast<int>(m_detections.size()))
        return QVariant();

    const Detection &det = m_detections[index.row()];

    switch (role) {
    case ClassIdRole:
        return det.classId;
    case ConfidenceRole:
        return det.confidence;
    case LabelRole:
        return det.label;
    case XRole:
        return det.x;
    case YRole:
        return det.y;
    case WRole:
        return det.w;
    case HRole:
        return det.h;
    case DataRole:
        return QVariant::fromValue(det);
    default:
        return QVariant();
    }
}

QHash<int, QByteArray> DetectionListModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[ClassIdRole] = "classId";
    roles[ConfidenceRole] = "confidence";
    roles[LabelRole] = "label";
    roles[XRole] = "x";
    roles[YRole] = "y";
    roles[WRole] = "w";
    roles[HRole] = "h";
    roles[DataRole] = "modelData"; // To keep compatibility with 'modelData' usage in QML
    return roles;
}

void DetectionListModel::updateDetections(const std::vector<DL_RESULT>& results, const std::vector<std::string>& classNames)
{
    // Fast path: if empty and was empty
    if (results.empty() && m_detections.empty()) return;
    
    // qDebug() << "DetectionListModel::updateDetections count:" << results.size();

    // Simplest strategy: full reset
    // More complex: diffing (not worth it for <100 items usually)
    beginResetModel();
    m_detections.clear();
    m_detections.reserve(results.size());

    for (const auto& res : results) {
        Detection det; 
        det.classId = res.classId;
        det.confidence = res.confidence;
        det.label = QString::fromStdString(classNames[res.classId]);
        
        float w = res.box.width;
        float h = res.box.height;
        
        det.x = res.box.x / (float)AppConfig::FrameWidth;
        det.y = res.box.y / (float)AppConfig::FrameHeight;
        det.w = w / (float)AppConfig::FrameWidth;
        det.h = h / (float)AppConfig::FrameHeight;
        
        m_detections.push_back(det);
    }
    endResetModel();
}
