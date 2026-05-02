#pragma once

#include <QAbstractListModel>
#include <QQmlEngine>
#include <QSize>
#include <vector>
#include "../domain/Detection.h"
#include "../domain/DetectionResult.h"

class DetectionListModel : public QAbstractListModel
{
    Q_OBJECT

public:
    enum DetectionRoles {
        ClassIdRole = Qt::UserRole + 1,
        ConfidenceRole,
        LabelRole,
        XRole,
        YRole,
        WRole,
        HRole,
        DataRole 
    };

    explicit DetectionListModel(QObject *parent = nullptr);

    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    void updateDetections(const std::vector<DetectionResult>& results, 
                          const std::vector<std::string>& classNames, 
                          const QSize& frameSize);
    
    const std::vector<Detection>& getDetections() const { return m_detections; }

private:
    std::vector<Detection> m_detections;
};
