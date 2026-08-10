#ifndef MULTICHANNELADAPTER_H
#define MULTICHANNELADAPTER_H

#include "BaseAdapter.h"
#include "VideoSource.h"
#include "YoloPostProcess.h"
#include "YoloV8PpuPostProcess.h"
#include "YoloFacePostProcess.h"
#include "YoloPosePostProcess.h"
#include <dxrt/dxrt_cxx_api.h>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <opencv2/core.hpp>

// Per-channel context
struct ChannelContext {
    int index;
    std::unique_ptr<VideoSource> source;
    cv::Mat resultFrame;  // This channel's rendered output
    std::mutex frameMutex;
    std::thread thread;
    std::atomic<bool> running{false};
    
    // Grid position
    int posX;
    int posY;
    int width;
    int height;
    
    // Detection results for this channel
    std::vector<DetectionBox> detections;

    // Source info
    QString primaryPath;
    QString primaryType;
    QString secondaryPath;
    QString secondaryType;
    std::atomic<bool> useSecondary{false};
    std::atomic<bool> reopenRequested{false};
    double fps{0.0};
};

class MultiChannelAdapter : public BaseAdapter {
    Q_OBJECT
public:
    explicit MultiChannelAdapter(QObject* parent = nullptr);
    ~MultiChannelAdapter() override;

    bool loadConfig(const QJsonObject& config) override;
    void start() override;
    void stop() override;

protected:
    void processLoop() override; // Composite frame emitter

private:
    void channelLoop(int channelIndex);
    void preprocess(const cv::Mat& img, std::vector<uint8_t>& buffer, float& scale, std::vector<int>& pad);

    std::shared_ptr<dxrt::InferenceEngine> m_engine;  // Shared across channels
    std::unique_ptr<YoloPostProcess> m_postProcess;          // For YOLOv5/v7/etc
    std::unique_ptr<YoloV8PpuPostProcess> m_postProcessV8;   // For YOLOv8 PPU
    std::unique_ptr<YoloFacePostProcess> m_postProcessFace;
    std::unique_ptr<YoloPosePostProcess> m_postProcessPose;

    enum ModelType {
        YOLO_STANDARD,
        YOLO_V8_PPU,
        YOLO_FACE,
        YOLO_POSE
    } m_modelType = YOLO_STANDARD;
    
    std::vector<std::unique_ptr<ChannelContext>> m_channels;
    
    int m_inputWidth;
    int m_inputHeight;
    int m_gridDim;        // ceil(sqrt(N))
    int m_compositeWidth;
    int m_compositeHeight;
    
    // Trespassing
    std::vector<cv::Point2f> m_trespassingPolygon;
    bool isPointInPolygon(const cv::Point2f& pt, const std::vector<cv::Point2f>& polygon);
    
    std::vector<std::string> m_classNames;
    
    // Composite frame
    cv::Mat m_compositeFrame;
    std::mutex m_compositeMutex;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_lastGlobalTime;
};

#endif // MULTICHANNELADAPTER_H
