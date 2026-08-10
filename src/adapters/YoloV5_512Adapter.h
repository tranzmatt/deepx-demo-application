#ifndef YOLOV5_512ADAPTER_H
#define YOLOV5_512ADAPTER_H

#include "BaseAdapter.h"
#include "VideoSource.h"
#include "YoloPostProcess.h"
#include <dxrt/dxrt_cxx_api.h>
#include <memory>

class YoloV5_512Adapter : public BaseAdapter {
    Q_OBJECT
public:
    explicit YoloV5_512Adapter(QObject* parent = nullptr);
    ~YoloV5_512Adapter() override;

    bool loadConfig(const QJsonObject& config) override;

protected:
    void processLoop() override;

private:
    std::unique_ptr<VideoSource> m_source;
    std::unique_ptr<dxrt::InferenceEngine> m_engine;
    std::unique_ptr<YoloPostProcess> m_postProcess;

    int m_inputWidth;
    int m_inputHeight;
    
    void preprocess(const cv::Mat& img, cv::Mat& preprocessed, float& scale, std::vector<int>& pad);
};

#endif // YOLOV5_512ADAPTER_H
