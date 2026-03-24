#ifndef HANDLANDMARKADAPTER_H
#define HANDLANDMARKADAPTER_H

#include "BaseAdapter.h"
#include "VideoSource.h"
#include "YoloHandPosePostProcess.h"
#include <dxrt/dxrt_api.h>
#include <memory>

class HandLandmarkAdapter : public BaseAdapter {
    Q_OBJECT
public:
    explicit HandLandmarkAdapter(QObject* parent = nullptr);
    ~HandLandmarkAdapter() override;

    bool loadConfig(const QJsonObject& config) override;

protected:
    void processLoop() override;

private:
    std::unique_ptr<VideoSource> m_source;
    std::unique_ptr<dxrt::InferenceEngine> m_engine;
    std::unique_ptr<YoloHandPosePostProcess> m_postProcess;

    int m_inputWidth;
    int m_inputHeight;
    
    void preprocess(const cv::Mat& img, cv::Mat& preprocessed, float& scale, std::vector<int>& pad);
};

#endif // HANDLANDMARKADAPTER_H
