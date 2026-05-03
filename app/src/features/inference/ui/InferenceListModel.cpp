#include "InferenceListModel.h"
#include <QVariant>
#include <QDebug>
#include <algorithm>

InferenceListModel::InferenceListModel(QObject *parent)
    : QAbstractListModel(parent)
{
}

int InferenceListModel::rowCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;
    return static_cast<int>(m_visionObjects.size());
}

QVariant InferenceListModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid() || index.row() >= static_cast<int>(m_visionObjects.size()))
        return QVariant();

    const VisionObject &det = m_visionObjects[index.row()];

    switch (role) {
    case ClassIdRole:
        return det.m_classId;
    case ConfidenceRole:
        return det.m_confidence;
    case LabelRole:
        return det.m_label;
    case XRole:
        return det.m_x;
    case YRole:
        return det.m_y;
    case WRole:
        return det.m_w;
    case HRole:
        return det.m_h;
    case DataRole:
        return QVariant::fromValue(det);
    default:
        return QVariant();
    }
}

QHash<int, QByteArray> InferenceListModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[ClassIdRole] = "classId";
    roles[ConfidenceRole] = "confidence";
    roles[LabelRole] = "label";
    roles[XRole] = "x";
    roles[YRole] = "y";
    roles[WRole] = "w";
    roles[HRole] = "h";
    roles[DataRole] = "modelData"; 
    return roles;
}

void InferenceListModel::updateResults(const std::vector<InferenceResult>& results, 
                                        const std::vector<std::string>& classNames, 
                                        const QSize& frameSize)
{
    if (results.empty() && m_visionObjects.empty() && m_frameSize == frameSize) return;
    
    if (m_frameSize != frameSize) {
        m_frameSize = frameSize;
        emit frameSizeChanged();
    }
    
    if (frameSize.isEmpty()) return;
    
    float frameW = static_cast<float>(frameSize.width());
    float frameH = static_cast<float>(frameSize.height());

    auto buildVisionObject = [&](const InferenceResult& res) -> VisionObject {
        VisionObject det; 
        det.m_classId = res.classId;
        det.m_confidence = res.confidence;
        if (res.classId >= 0 && res.classId < classNames.size()) {
            det.m_label = QString::fromStdString(classNames[res.classId]);
        } else {
            det.m_label = QString("Class %1").arg(res.classId);
        }
        
        det.m_x = res.box.x / frameW;
        det.m_y = res.box.y / frameH;
        det.m_w = res.box.width / frameW;
        det.m_h = res.box.height / frameH;
        
        for (const auto& kp : res.keyPoints) {
            det.m_keyPoints.append(QPointF(kp.x / frameW, kp.y / frameH));
        }
        
        return det;
    };

    int oldSize = m_visionObjects.size();
    int newSize = results.size();

    int updateCount = std::min(oldSize, newSize);
    for (int i = 0; i < updateCount; ++i) {
        m_visionObjects[i] = buildVisionObject(results[i]);
        emit dataChanged(index(i), index(i));
    }

    if (newSize > oldSize) {
        beginInsertRows(QModelIndex(), oldSize, newSize - 1);
        for (int i = oldSize; i < newSize; ++i)
            m_visionObjects.push_back(buildVisionObject(results[i]));
        endInsertRows();
    }
    else if (newSize < oldSize) {
        beginRemoveRows(QModelIndex(), newSize, oldSize - 1);
        m_visionObjects.resize(newSize);
        endRemoveRows();
    }
}
