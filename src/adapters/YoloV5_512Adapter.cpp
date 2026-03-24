#include "YoloV5_512Adapter.h"
#include <QJsonArray>
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>
#include <opencv2/imgproc.hpp>
#include <chrono>

YoloV5_512Adapter::YoloV5_512Adapter(QObject* parent) : BaseAdapter(parent) {
    m_source = std::make_unique<VideoSource>();
}

YoloV5_512Adapter::~YoloV5_512Adapter() {
    stop();
}

bool YoloV5_512Adapter::loadConfig(const QJsonObject& config) {
    QJsonObject modelObj = config["model"].toObject();
    QString modelPath = modelObj["path"].toString();
    

    std::string resolvedModelPath = resolvePath(modelPath);
    qDebug() << "[YoloV5_512] Loading model from:" << QString::fromStdString(resolvedModelPath);

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

    QJsonObject modelParam = modelObj["param"].toObject();
    m_inputWidth = modelParam["input_width"].toInt(512);
    m_inputHeight = modelParam["input_height"].toInt(512);
    
    float scoreThresh = modelParam["score_threshold"].toDouble(0.3);
    float iouThresh = modelParam["iou_threshold"].toDouble(0.45);
    
    m_postProcess = std::make_unique<YoloPostProcess>(m_inputWidth, m_inputHeight, 0.25f, scoreThresh, iouThresh);

    QJsonObject outputObj = config["output"].toObject();
    QJsonArray classesArr = outputObj["classes"].toArray();
    std::vector<std::string> classNames;
    for (const auto& cls : classesArr) {
        classNames.push_back(cls.toString().toStdString());
    }
    m_postProcess->setClassNames(classNames);

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

void YoloV5_512Adapter::preprocess(const cv::Mat& img, cv::Mat& preprocessed, float& scale, std::vector<int>& pad) {
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

void YoloV5_512Adapter::processLoop() {
    cv::Mat frame;
    cv::Mat preprocessed;
    
    while (m_running) {
        if (m_reopenRequested) {
            m_reopenRequested = false;
            m_source->close();
            SourceInfo target = getCurrentSource();
            if (!openSource(target, m_source.get())) {
                 qWarning() << "[YoloV5_512] Failed to switch to source:" << target.path;
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
        
        auto tInfer = std::chrono::high_resolution_clock::now();
        std::vector<DetectionBox> detections = m_postProcess->postprocess(outputs, m_inputWidth, m_inputHeight);

        // Adjust coordinates and apply Blur for "person"
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

            // Privacy Masking Logic: Blur "person"
            if (box.label == "person") {
                int bx = (int)(box.box.x * frame.cols);
                int by = (int)(box.box.y * frame.rows);
                int bw = (int)(box.box.width * frame.cols);
                int bh = (int)(box.box.height * frame.rows);

                // Clamp to frame boundaries
                bx = std::max(0, bx);
                by = std::max(0, by);
                bw = std::min(bw, frame.cols - bx);
                bh = std::min(bh, frame.rows - by);

                if (bw > 0 && bh > 0) {
                    cv::Rect roiRect(bx, by, bw, bh);
                    cv::Mat roi = frame(roiRect);
                    // Use fast box blur for minimal CPU usage
                    cv::blur(roi, roi, cv::Size(100, 100)); 
                }
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
