#include "YoloFaceAdapter.h"
#include <QDebug>
#include <QJsonArray>
#include <QJsonObject>
#include <QFileInfo>
#include <QCoreApplication>
#include <QDir>
#include <opencv2/imgproc.hpp>

YoloFaceAdapter::YoloFaceAdapter(QObject* parent) : YoloAdapter(parent) {
}

bool YoloFaceAdapter::loadConfig(const QJsonObject& config) {
    // 1. Load model
    QJsonObject modelObj = config["model"].toObject();
    QString modelPath = modelObj["path"].toString();
    

    std::string resolvedModelPath = resolvePath(modelPath);
    qDebug() << "[YoloFaceAdapter] Loading model from:" << QString::fromStdString(resolvedModelPath);

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

    // 2. Setup postprocess
    QJsonObject modelParam = modelObj["param"].toObject();
    m_inputWidth = modelParam["input_width"].toInt(640);
    m_inputHeight = modelParam["input_height"].toInt(640);
    
    float scoreThresh = modelParam["score_threshold"].toDouble(0.25);
    float iouThresh = modelParam["iou_threshold"].toDouble(0.45);
    
    m_facePostProcess = std::make_unique<YoloFacePostProcess>(m_inputWidth, m_inputHeight, scoreThresh, iouThresh);

    // 3. Open source
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

void YoloFaceAdapter::processLoop() {
    cv::Mat frame;
    cv::Mat preprocessed;
    
    while (m_running) {
        if (m_reopenRequested) {
            m_reopenRequested = false;
            m_source->close();
            SourceInfo target = getCurrentSource();
            if (!openSource(target, m_source.get())) {
                 qWarning() << "[YoloFace] Failed to switch to source:" << target.path;
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
        // Reuse base preprocess
        preprocess(frame, preprocessed, scale, pad);

        // Run Inference
        auto reqId = m_engine->RunAsync(preprocessed.data, nullptr, nullptr);
        auto outputs = m_engine->Wait(reqId);
        
        // Postprocess
        auto tInfer = std::chrono::high_resolution_clock::now();
        
        std::vector<DetectionBox> detections = m_facePostProcess->postprocess(outputs, m_inputWidth, m_inputHeight);

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
            
            box.box.x = x / frame.cols;
            box.box.y = y / frame.rows;
            box.box.width = w / frame.cols;
            box.box.height = h / frame.rows;
            
            // Adjust landmarks
            for (size_t k = 0; k < box.landmarks.size(); k += 3) {
                float lx = box.landmarks[k] * m_inputWidth;
                float ly = box.landmarks[k+1] * m_inputHeight;
                
                lx = (lx - pad[0]) / scale;
                ly = (ly - pad[1]) / scale;
                
                box.landmarks[k] = lx / frame.cols;
                box.landmarks[k+1] = ly / frame.rows;
            }
        }

        DetectionResult result;
        result.detections = detections;
        result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(tStart.time_since_epoch()).count();
        
        auto tEnd = std::chrono::high_resolution_clock::now();
        result.inferenceTime = std::chrono::duration<double, std::milli>(tInfer - tStart).count();
        double totalTime = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        QImage qimg((const uchar*)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        emit frameReady(qimg.copy(), result);
        
        emit fpsUpdate(1000.0 / totalTime);
    }
}
