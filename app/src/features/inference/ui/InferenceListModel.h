#pragma once

#include <QAbstractListModel>
#include <QQmlEngine>
#include <QSize>
#include <vector>
#include "../domain/VisionObject.h"
#include "../domain/InferenceResult.h"

class InferenceListModel : public QAbstractListModel
{
    Q_OBJECT

    Q_PROPERTY(QSize frameSize READ frameSize NOTIFY frameSizeChanged)

public:
    enum InferenceRoles {
        ClassIdRole = Qt::UserRole + 1,
        ConfidenceRole,
        LabelRole,
        XRole,
        YRole,
        WRole,
        HRole,
        DataRole 
    };

    explicit InferenceListModel(QObject *parent = nullptr);

    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    void updateResults(const std::vector<InferenceResult>& results, 
                          const std::vector<std::string>& classNames, 
                          const QSize& frameSize);
    
    const std::vector<VisionObject>& getVisionObjects() const { return m_visionObjects; }
    QSize frameSize() const { return m_frameSize; }

signals:
    void frameSizeChanged();

private:
    std::vector<VisionObject> m_visionObjects;
    QSize m_frameSize;
};
