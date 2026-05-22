#include <gtest/gtest.h>
#include "InferenceListModel.h"
#include <QSize>

TEST(InferenceListModelTest, EmptyModel) {
    InferenceListModel model;
    EXPECT_EQ(model.rowCount(), 0);
}

TEST(InferenceListModelTest, UpdateResults) {
    InferenceListModel model;
    std::vector<InferenceResult> results;
    
    InferenceResult res;
    res.classId = 0;
    res.confidence = 0.95f;
    res.box = cv::Rect(100, 100, 200, 200);
    results.push_back(res);

    std::vector<std::string> classes = {"person", "car"};
    QSize frameSize(1000, 1000);

    model.updateResults(results, classes, frameSize);

    ASSERT_EQ(model.rowCount(), 1);
    
    // Check data roles
    QModelIndex idx = model.index(0);
    EXPECT_EQ(model.data(idx, InferenceListModel::ClassIdRole).toInt(), 0);
    EXPECT_NEAR(model.data(idx, InferenceListModel::ConfidenceRole).toFloat(), 0.95f, 1e-5);
    EXPECT_EQ(model.data(idx, InferenceListModel::LabelRole).toString(), "person");
    
    // Normalized coordinates
    EXPECT_NEAR(model.data(idx, InferenceListModel::XRole).toFloat(), 0.1f, 1e-5); // 100/1000
    EXPECT_NEAR(model.data(idx, InferenceListModel::YRole).toFloat(), 0.1f, 1e-5);
    EXPECT_NEAR(model.data(idx, InferenceListModel::WRole).toFloat(), 0.2f, 1e-5);
    EXPECT_NEAR(model.data(idx, InferenceListModel::HRole).toFloat(), 0.2f, 1e-5);
}

TEST(InferenceListModelTest, RowManagement) {
    InferenceListModel model;
    std::vector<std::string> classes = {"person"};
    QSize frameSize(640, 480);

    // Add 2 results
    std::vector<InferenceResult> results(2);
    results[0].classId = 0;
    results[1].classId = 0;
    model.updateResults(results, classes, frameSize);
    EXPECT_EQ(model.rowCount(), 2);

    // Update to 1 result
    results.resize(1);
    model.updateResults(results, classes, frameSize);
    EXPECT_EQ(model.rowCount(), 1);

    // Update to 0 results
    results.clear();
    model.updateResults(results, classes, frameSize);
    EXPECT_EQ(model.rowCount(), 0);
}
