#ifndef OCRADAPTER_H
#define OCRADAPTER_H

#include "BaseAdapter.h"
#include "VideoSource.h"
#include "OcrPostProcess.h"
#include <dxrt/dxrt_api.h>
#include <map>
#include <memory>

class OcrAdapter : public BaseAdapter {
    Q_OBJECT
public:
    explicit OcrAdapter(QObject* parent = nullptr);
    ~OcrAdapter() override;

    bool loadConfig(const QJsonObject& config) override;
    void handleMouseEvent(int x, int y, int type) override;

protected:
    void processLoop() override;

private:
    std::unique_ptr<VideoSource> m_source;
    
    // --- Inference Engines ---
    std::unique_ptr<dxrt::InferenceEngine> m_detEngine;
    std::unique_ptr<dxrt::InferenceEngine> m_clsEngine;
    // Map: aspect_ratio -> height -> engine
    std::map<int, std::map<int, std::shared_ptr<dxrt::InferenceEngine>>> m_recEngines;
    
    // --- Post Processing ---
    std::unique_ptr<OcrPostProcess> m_postProcess;
    
    // --- Model Parameters ---
    // Detection
    int m_detInputSize = 640;
    
    // Classification
    int m_clsInputHeight = 48;
    int m_clsInputWidth = 192;
    float m_clsThresh = 0.9f;
    
    // Recognition
    int m_recInputHeight = 48; // Base height
    // Width is dynamic based on ratio
    
    // --- Interactive ---
    std::vector<QString> m_imagePaths;
    std::vector<cv::Mat> m_thumbnails;
    int m_selectedImageIndex = -1;
    bool m_needsRecognition = false;
    bool m_isStream = false;
    QImage m_lastAnnotated;
    QImage m_lastTextPanel;
    std::vector<QRect> m_thumbnailRects;
    std::mutex m_stateMutex;
    
    // --- Helper Methods ---
    // Preprocess image for detection (resize, pad, normalize)
    void preprocessDet(const cv::Mat& img, cv::Mat& out, float& scale, std::vector<int>& pad);
    
    // Get rotated crop based on box points
    cv::Mat getRotateCrop(const cv::Mat& img, const std::vector<cv::Point>& box);
    
    // Select appropriate recognition model based on crop dimensions
    std::shared_ptr<dxrt::InferenceEngine> getRecModel(int width, int height);
    
    // Resize for recognition model
    void preprocessRec(const cv::Mat& img, cv::Mat& out, int targetHeight, int targetWidth);

    // Preprocess for classification
    void preprocessCls(const cv::Mat& img, cv::Mat& out);

    // Drawing helpers
    QImage drawOcrResults(const cv::Mat& img, const DetectionResult& result);
    QImage createTextPanel(const DetectionResult& result, int h, int w);
};

#endif // OCRADAPTER_H
