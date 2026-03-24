#ifndef YOLOPOSEADAPTER_H
#define YOLOPOSEADAPTER_H

#include "BaseAdapter.h"
#include "VideoSource.h"
#include "YoloPosePostProcess.h"
#include <dxrt/dxrt_api.h>
#include <memory>

class YoloPoseAdapter : public BaseAdapter {
    Q_OBJECT
public:
    explicit YoloPoseAdapter(QObject* parent = nullptr);
    ~YoloPoseAdapter() override;

    bool loadConfig(const QJsonObject& config) override;

protected:
    void processLoop() override;

private:
    std::unique_ptr<VideoSource> m_source;
    std::unique_ptr<dxrt::InferenceEngine> m_engine;
    std::unique_ptr<YoloPosePostProcess> m_postProcess;

    int m_inputWidth;
    int m_inputHeight;
    std::vector<int> m_padXy;
    
    void preprocess(const cv::Mat& img, cv::Mat& preprocessed, float& scale, std::vector<int>& pad);
};

#endif // YOLOPOSEADAPTER_H
