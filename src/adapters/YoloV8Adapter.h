#ifndef YOLOV8ADAPTER_H
#define YOLOV8ADAPTER_H

#include "BaseAdapter.h"
#include "VideoSource.h"
#include "YoloV8PpuPostProcess.h"
#include <dxrt/dxrt_cxx_api.h>
#include <memory>

class YoloV8Adapter : public BaseAdapter {
    Q_OBJECT
public:
    explicit YoloV8Adapter(QObject* parent = nullptr);
    ~YoloV8Adapter() override;

    bool loadConfig(const QJsonObject& config) override;

protected:
    void processLoop() override;
    virtual void onResultReady(DetectionResult& result) {}

private:
    std::unique_ptr<VideoSource> m_source;
    std::unique_ptr<dxrt::InferenceEngine> m_engine;
    std::unique_ptr<YoloV8PpuPostProcess> m_postProcess;

    int m_inputWidth;
    int m_inputHeight;
    std::vector<std::string> m_classNames;
    
    void preprocess(const cv::Mat& img, cv::Mat& preprocessed, float& scale, std::vector<int>& pad);
};

#endif // YOLOV8ADAPTER_H
