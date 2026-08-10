#ifndef FACEALIGNMENTADAPTER_H
#define FACEALIGNMENTADAPTER_H

#include "BaseAdapter.h"
#include "VideoSource.h"
#include "YoloFacePostProcess.h"
#include <dxrt/dxrt_cxx_api.h>
#include <memory>

class FaceAlignmentAdapter : public BaseAdapter {
    Q_OBJECT
public:
    explicit FaceAlignmentAdapter(QObject* parent = nullptr);
    ~FaceAlignmentAdapter() override;

    bool loadConfig(const QJsonObject& config) override;

protected:
    void processLoop() override;

private:
    std::unique_ptr<VideoSource> m_source;
    std::unique_ptr<dxrt::InferenceEngine> m_engine;
    std::unique_ptr<dxrt::InferenceEngine> m_emotionEngine;
    std::unique_ptr<YoloFacePostProcess> m_postProcess;

    int m_inputWidth;
    int m_inputHeight;
    int m_emotionInputWidth = 224;
    int m_emotionInputHeight = 224;
    std::vector<std::string> m_emotionClasses;

    void preprocess(const cv::Mat& img, cv::Mat& preprocessed, float& scale, std::vector<int>& pad);
};

#endif // FACEALIGNMENTADAPTER_H
