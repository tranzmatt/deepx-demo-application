#ifndef YOLOADAPTER_H
#define YOLOADAPTER_H

#include "BaseAdapter.h"
#include "VideoSource.h"
#include "YoloPostProcess.h"
#include <dxrt/dxrt_cxx_api.h>
#include <memory>

class YoloAdapter : public BaseAdapter {
    Q_OBJECT
public:
    explicit YoloAdapter(QObject* parent = nullptr);
    ~YoloAdapter() override;

    bool loadConfig(const QJsonObject& config) override;

protected:
    void processLoop() override;
    virtual void onResultReady(DetectionResult& result) {}

protected:
    std::unique_ptr<VideoSource> m_source;
    std::unique_ptr<dxrt::InferenceEngine> m_engine;
    std::unique_ptr<YoloPostProcess> m_postProcess;

    int m_inputWidth;
    int m_inputHeight;

    // Buffers
    std::vector<uint8_t> m_inputBuffer;
    
    void preprocess(const cv::Mat& img, cv::Mat& preprocessed, float& scale, std::vector<int>& pad);
};

#endif // YOLOADAPTER_H
