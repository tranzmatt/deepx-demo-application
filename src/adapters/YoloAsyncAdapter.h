#ifndef YOLOASYNCADAPTER_H
#define YOLOASYNCADAPTER_H

#include "BaseAdapter.h"
#include "VideoSource.h"
#include "YoloV8PpuPostProcess.h" // Uses V8 PostProcessing since configs point to V8 PPU model
#include <dxrt/dxrt_cxx_api.h>
#include <memory>
#include <deque>

class YoloAsyncAdapter : public BaseAdapter {
    Q_OBJECT
public:
    explicit YoloAsyncAdapter(QObject* parent = nullptr);
    ~YoloAsyncAdapter() override;

    bool loadConfig(const QJsonObject& config) override;

protected:
    virtual void processLoop() override;
    
    // Hook for derived classes to modify logic if needed
    virtual void onResultReady(DetectionResult& result);

protected:
    std::unique_ptr<VideoSource> m_source;
    std::unique_ptr<dxrt::InferenceEngine> m_engine;
    std::unique_ptr<YoloV8PpuPostProcess> m_postProcess;

    int m_inputWidth;
    int m_inputHeight;
    std::vector<std::string> m_classNames;
    
    struct RequestContext {
        int reqId;
        cv::Mat originalFrame;
        float scale;
        std::vector<int> pad;
        long long timestamp;
        std::chrono::time_point<std::chrono::high_resolution_clock> tStart;
        std::vector<uint8_t> inputBuffer; // Keep buffer alive!
    };
    
    void preprocess(const cv::Mat& img, std::vector<uint8_t>& buffer, float& scale, std::vector<int>& pad);
};

#endif // YOLOASYNCADAPTER_H
