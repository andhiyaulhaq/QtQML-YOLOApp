#ifndef DETECTIONLISTMODEL_H
#define DETECTIONLISTMODEL_H

#include <QAbstractListModel>
#include <QVideoSink>
#include <QQmlEngine>
#include "DetectionStruct.h"
#include "inference.h" 

class DetectionListModel : public QAbstractListModel
{
    Q_OBJECT
    QML_ELEMENT

public:
    enum DetectionRoles {
        ClassIdRole = Qt::UserRole + 1,
        ConfidenceRole,
        LabelRole,
        XRole,
        YRole,
        WRole,
        HRole,
        DataRole // Full detection object
    };

    explicit DetectionListModel(QObject *parent = nullptr);

    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    // Fast update method called by VideoController
    void updateDetections(const std::vector<DL_RESULT>& results, const std::vector<std::string>& classNames);
    
    const std::vector<Detection>& getDetections() const { return m_detections; }

private:
    std::vector<Detection> m_detections;
};

#endif // DETECTIONLISTMODEL_H
