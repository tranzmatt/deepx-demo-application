#include "YoloV8Adapter.h"
#include "Utils.h"
#include <QJsonArray>
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>
#include <opencv2/imgproc.hpp>
#include <chrono>

YoloV8Adapter::YoloV8Adapter(QObject* parent) : BaseAdapter(parent) {
    m_source = std::make_unique<VideoSource>();
}

YoloV8Adapter::~YoloV8Adapter() {
    stop();
}

bool YoloV8Adapter::loadConfig(const QJsonObject& config) {
    // 1. Load Model
    QJsonObject modelObj = config["model"].toObject();
    QString modelPath = modelObj["path"].toString();
    

    std::string resolvedModelPath = resolvePath(modelPath);

    dxrt::InferenceOption io;
    try {
        m_engine = std::make_unique<dxrt::InferenceEngine>(resolvedModelPath, io);
    } catch (const dxrt::Exception& e) {
        emit errorOccurred(QString("Failed to load model: %1").arg(e.what()));
        return false;
    } catch (const std::exception& e) {
        emit errorOccurred(QString("Failed to load model: %1").arg(e.what()));
        return false;
    } catch (...) {
        emit errorOccurred("Unknown exception while loading model");
        return false;
    }

    // 2. Setup PostProcess
    QJsonObject modelParam = modelObj["param"].toObject();
    m_inputWidth = modelParam["input_width"].toInt(640);
    m_inputHeight = modelParam["input_height"].toInt(640);
    float scoreThresh = modelParam["score_threshold"].toDouble(0.25);
    float iouThresh = modelParam["iou_threshold"].toDouble(0.45);
    
    m_postProcess = std::make_unique<YoloV8PpuPostProcess>(m_inputWidth, m_inputHeight, scoreThresh, iouThresh);

    // Load Classes
    QJsonObject outputObj = config["output"].toObject();
    QJsonArray classesArr = outputObj["classes"].toArray();
    for (const auto& cls : classesArr) {
        m_classNames.push_back(cls.toString().toStdString());
    }

    // 3. Open Source
    QJsonObject inputObj = config["input"].toObject();
    QJsonArray sources = inputObj["sources"].toArray();
    if (sources.isEmpty()) {
        emit errorOccurred("No sources");
        return false;
    }
    
    m_sources.clear();
    for (const auto& srcVal : sources) {
        QJsonObject srcObj = srcVal.toObject();
        SourceInfo info;
        info.path = srcObj["path"].toString();
        info.type = srcObj["type"].toString();
        info.flip = srcObj["flip"].toBool(false);
        m_sources.push_back(info);
    }
    
    QJsonObject firstSource = sources[0].toObject();
    if (firstSource.contains("secondary_path") && m_sources.size() == 1) {
        SourceInfo secInfo;
        secInfo.path = firstSource["secondary_path"].toString(m_sources[0].path);
        secInfo.type = firstSource["secondary_type"].toString(m_sources[0].type);
        secInfo.flip = firstSource["flip"].toBool(false);
        m_sources.push_back(secInfo);
    }

    if (!openSource(m_sources[0], m_source.get())) {
         emit errorOccurred(QString("Failed to open source: %1").arg(m_sources[0].path));
         return false;
    }
    
    return true;
}

void YoloV8Adapter::preprocess(const cv::Mat& img, cv::Mat& preprocessed, float& scale, std::vector<int>& pad) {
    int w = img.cols;
    int h = img.rows;
    scale = std::min((float)m_inputWidth / w, (float)m_inputHeight / h);
    int newUnpadW = (int)round(w * scale);
    int newUnpadH = (int)round(h * scale);
    int dw = (m_inputWidth - newUnpadW) / 2;
    int dh = (m_inputHeight - newUnpadH) / 2;
    pad = {dw, dh};
    cv::Mat resized;
    if (w != newUnpadW || h != newUnpadH) {
        cv::resize(img, resized, cv::Size(newUnpadW, newUnpadH));
    } else {
        resized = img;
    }
    cv::copyMakeBorder(resized, preprocessed, dh, m_inputHeight - newUnpadH - dh, dw, m_inputWidth - newUnpadW - dw, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::cvtColor(preprocessed, preprocessed, cv::COLOR_BGR2RGB);
}

void YoloV8Adapter::processLoop() {
    cv::Mat frame;
    cv::Mat preprocessed;

    // Pipelined inference: submit frame N, display frame N with result from N-1
    DetectionResult lastResult;
    bool hasPending = false;
    int pendingReqId = -1;
    float pendingScale = 1.0f;
    std::vector<int> pendingPad;
    int pendingFrameCols = 0;
    int pendingFrameRows = 0;
    auto tPendingStart = std::chrono::high_resolution_clock::now();

    while (m_running) {
        if (m_reopenRequested) {
            m_reopenRequested = false;
            m_source->close();
            SourceInfo target = getCurrentSource();
            if (!openSource(target, m_source.get())) {
                 qWarning() << "[YoloV8] Failed to switch to source:" << target.path;
            }
            hasPending = false;
        }

        auto tStart = std::chrono::high_resolution_clock::now();
        if (!m_source->read(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        if (frame.empty()) continue;

        // Collect result from previous async inference (if any)
        if (hasPending) {
            auto outputs = m_engine->Wait(pendingReqId);
            auto tInfer = std::chrono::high_resolution_clock::now();
            std::vector<DetectionBox> detections = m_postProcess->postprocess(outputs, m_inputWidth, m_inputHeight);

            for (auto& box : detections) {
                float x = box.box.x * m_inputWidth;
                float y = box.box.y * m_inputHeight;
                float w = box.box.width * m_inputWidth;
                float h = box.box.height * m_inputHeight;

                x = (x - pendingPad[0]) / pendingScale;
                y = (y - pendingPad[1]) / pendingScale;
                w = w / pendingScale;
                h = h / pendingScale;

                box.box.x = x / pendingFrameCols;
                box.box.y = y / pendingFrameRows;
                box.box.width = w / pendingFrameCols;
                box.box.height = h / pendingFrameRows;

                if (box.classId < m_classNames.size()) {
                    box.label = m_classNames[box.classId];
                } else {
                    box.label = "Unknown";
                }
            }

            lastResult.detections = detections;
            lastResult.inferenceTime = std::chrono::duration<double, std::milli>(tInfer - tPendingStart).count();
            onResultReady(lastResult);
            hasPending = false;
        }

        // Submit current frame for async inference
        float scale;
        std::vector<int> pad;
        preprocess(frame, preprocessed, scale, pad);
        pendingReqId = m_engine->RunAsync(preprocessed.data, nullptr, nullptr);
        pendingScale = scale;
        pendingPad = pad;
        pendingFrameCols = frame.cols;
        pendingFrameRows = frame.rows;
        tPendingStart = tStart;
        hasPending = true;

        // Display current frame immediately with last known result
        lastResult.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(tStart.time_since_epoch()).count();

        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        QImage qimg((const uchar*)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);

        emit frameReady(qimg.copy(), lastResult);

        auto tEnd = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        emit fpsUpdate(1000.0 / totalTime);
    }

    // Drain pending inference on exit
    if (hasPending) {
        try { m_engine->Wait(pendingReqId); } catch (...) {}
    }
}
