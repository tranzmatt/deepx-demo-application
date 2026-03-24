#include "YoloAdapter.h"
#include <QJsonArray>
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>
#include <opencv2/imgproc.hpp>
#include <chrono>

YoloAdapter::YoloAdapter(QObject* parent) : BaseAdapter(parent) {
    m_source = std::make_unique<VideoSource>();
}

YoloAdapter::~YoloAdapter() {
    stop();
}

bool YoloAdapter::loadConfig(const QJsonObject& config) {
    // 1. Load Model
    QJsonObject modelObj = config["model"].toObject();
    QString modelPath = modelObj["path"].toString();
    

    std::string resolvedModelPath = resolvePath(modelPath);
    qDebug() << "Loading model from:" << QString::fromStdString(resolvedModelPath);

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
    
    float scoreThresh = modelParam["score_threshold"].toDouble(0.3);
    float iouThresh = modelParam["iou_threshold"].toDouble(0.45);
    
    m_postProcess = std::make_unique<YoloPostProcess>(m_inputWidth, m_inputHeight, 0.25f, scoreThresh, iouThresh);

    // Load classes
    QJsonObject outputObj = config["output"].toObject();
    QJsonArray classesArr = outputObj["classes"].toArray();
    std::vector<std::string> classNames;
    for (const auto& cls : classesArr) {
        classNames.push_back(cls.toString().toStdString());
    }
    m_postProcess->setClassNames(classNames);

    // 3. Open Source
    QJsonObject inputObj = config["input"].toObject();
    QJsonArray sources = inputObj["sources"].toArray();
    if (sources.isEmpty()) {
        emit errorOccurred("No sources defined in config");
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
    
    // Support legacy secondary_path in first object
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

void YoloAdapter::preprocess(const cv::Mat& img, cv::Mat& preprocessed, float& scale, std::vector<int>& pad) {
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
     // BGR to RGB if model expects it? Usually YoloV5 in dxrt demos uses RGB.
    cv::cvtColor(preprocessed, preprocessed, cv::COLOR_BGR2RGB);
}

void YoloAdapter::processLoop() {
    cv::Mat frame;
    cv::Mat preprocessed;
    
    while (m_running) {
        // Handle source re-opening
        if (m_reopenRequested) {
            std::lock_guard<std::mutex> lock(m_sourceMutex);
            m_reopenRequested = false;
            m_source->close();
            
            SourceInfo target = getCurrentSource();
            
            if (!openSource(target, m_source.get())) {
                qWarning() << "[YoloAdapter] Failed to switch to source:" << target.path;
            } else {
                qDebug() << "[YoloAdapter] Successfully switched to source:" << target.path;
            }
        }

        auto tStart = std::chrono::high_resolution_clock::now();
        
        if (!m_source->read(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        if (frame.empty()) continue;

        float scale;
        std::vector<int> pad;
        preprocess(frame, preprocessed, scale, pad);

        auto reqId = m_engine->RunAsync(preprocessed.data, nullptr, nullptr);
        auto outputs = m_engine->Wait(reqId);
        
        // Postprocess
        auto tInfer = std::chrono::high_resolution_clock::now();
        std::vector<DetectionBox> detections = m_postProcess->postprocess(outputs, m_inputWidth, m_inputHeight);

        // Adjust coordinates back to original image
        for (auto& box : detections) {
            float x = box.box.x * m_inputWidth;
            float y = box.box.y * m_inputHeight;
            float w = box.box.width * m_inputWidth;
            float h = box.box.height * m_inputHeight;

            x = (x - pad[0]) / scale;
            y = (y - pad[1]) / scale;
            w = w / scale;
            h = h / scale;
            
            // Normalize to original frame
            box.box.x = x / frame.cols;
            box.box.y = y / frame.rows;
            box.box.width = w / frame.cols;
            box.box.height = h / frame.rows;
        }

        // Emit result
        DetectionResult result;
        result.detections = detections;
        result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(tStart.time_since_epoch()).count();
        
        auto tEnd = std::chrono::high_resolution_clock::now();
        result.inferenceTime = std::chrono::duration<double, std::milli>(tInfer - tStart).count();
        double totalTime = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        QImage qimg((const uchar*)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        onResultReady(result);
        emit frameReady(qimg.copy(), result); // .copy() ensures it owns data
        
        emit fpsUpdate(1000.0 / totalTime);
    }
}
