#include "YoloAsyncAdapter.h"
#include <QJsonArray>
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>
#include <opencv2/imgproc.hpp>
#include <chrono>

YoloAsyncAdapter::YoloAsyncAdapter(QObject* parent) : BaseAdapter(parent) {
    m_source = std::make_unique<VideoSource>();
}

YoloAsyncAdapter::~YoloAsyncAdapter() {
    stop();
}

bool YoloAsyncAdapter::loadConfig(const QJsonObject& config) {
    QJsonObject modelObj = config["model"].toObject();
    QString modelPath = modelObj["path"].toString();
    

    std::string resolvedModelPath = resolvePath(modelPath);
    qDebug() << "YoloAsyncAdapter::loadConfig() - resolved model path:" << QString::fromStdString(resolvedModelPath);

    dxrt::InferenceOption io;
    try {
        m_engine = std::make_unique<dxrt::InferenceEngine>(resolvedModelPath, io);
    } catch (const dxrt::Exception& e) {
        qCritical() << "YoloAsyncAdapter::loadConfig() - ERROR creating InferenceEngine:" << e.what();
        emit errorOccurred(QString("Failed to load Async model: %1").arg(e.what()));
        return false;
    } catch (const std::exception& e) {
        qCritical() << "YoloAsyncAdapter::loadConfig() - ERROR creating InferenceEngine:" << e.what();
        emit errorOccurred(QString("Failed to load Async model: %1").arg(e.what()));
        return false;
    }

    QJsonObject modelParam = modelObj["param"].toObject();
    m_inputWidth = modelParam["input_width"].toInt(640);
    m_inputHeight = modelParam["input_height"].toInt(640);
    float scoreThresh = modelParam["score_threshold"].toDouble(0.25);
    float iouThresh = modelParam["iou_threshold"].toDouble(0.45);
    
    m_postProcess = std::make_unique<YoloV8PpuPostProcess>(m_inputWidth, m_inputHeight, scoreThresh, iouThresh);

    QJsonObject outputObj = config["output"].toObject();
    QJsonArray classesArr = outputObj["classes"].toArray();
    for (const auto& cls : classesArr) {
        m_classNames.push_back(cls.toString().toStdString());
    }

    QJsonObject inputObj = config["input"].toObject();
    QJsonArray sources = inputObj["sources"].toArray();
    if (sources.isEmpty()) {
        qWarning() << "YoloAsyncAdapter::loadConfig() - sources empty";
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
         qWarning() << "YoloAsyncAdapter::loadConfig() - failed to open source";
         emit errorOccurred(QString("Failed to open source: %1").arg(m_sources[0].path));
         return false;
    }
    
    return true;
}

void YoloAsyncAdapter::preprocess(const cv::Mat& img, std::vector<uint8_t>& buffer, float& scale, std::vector<int>& pad) {
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
    
    cv::Mat preprocessed;
    cv::copyMakeBorder(resized, preprocessed, dh, m_inputHeight - newUnpadH - dh, dw, m_inputWidth - newUnpadW - dw, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::cvtColor(preprocessed, preprocessed, cv::COLOR_BGR2RGB);

    // Copy to persistent buffer
    size_t size = preprocessed.total() * preprocessed.elemSize();
    buffer.resize(size);
    memcpy(buffer.data(), preprocessed.data, size);
}

void YoloAsyncAdapter::processLoop() {
    cv::Mat frame;
    
    RequestContext pendingCtx;
    bool hasPending = false;
    
    while (m_running) {
        try {
            if (m_reopenRequested) {
                m_reopenRequested = false;
                m_source->close();
                SourceInfo target = getCurrentSource();
                if (!openSource(target, m_source.get())) {
                    qWarning() << "[YoloAsync] Failed to switch to source:" << target.path;
                }
                hasPending = false; // Reset pipeline on source switch
            }

            // 1. Wait for Pending (if any) before submitting next
            if (hasPending) {
                try {
                    auto outputs = m_engine->Wait(pendingCtx.reqId);
                    
                    // Postprocess Pending
                    auto tInfer = std::chrono::high_resolution_clock::now();
                    std::vector<DetectionBox> detections = m_postProcess->postprocess(outputs, m_inputWidth, m_inputHeight);
                    
                    // Adjust Coords
                    for (auto& box : detections) {
                        float x = box.box.x * m_inputWidth;
                        float y = box.box.y * m_inputHeight;
                        float w = box.box.width * m_inputWidth;
                        float h = box.box.height * m_inputHeight;

                        x = (x - pendingCtx.pad[0]) / pendingCtx.scale;
                        y = (y - pendingCtx.pad[1]) / pendingCtx.scale;
                        w = w / pendingCtx.scale;
                        h = h / pendingCtx.scale;
                        
                        box.box.x = x / pendingCtx.originalFrame.cols;
                        box.box.y = y / pendingCtx.originalFrame.rows;
                        box.box.width = w / pendingCtx.originalFrame.cols;
                        box.box.height = h / pendingCtx.originalFrame.rows;
                        
                        if (box.classId < m_classNames.size()) {
                            box.label = m_classNames[box.classId];
                        } else {
                            box.label = "Unknown";
                        }
                    }
                    
                    DetectionResult result;
                    result.detections = detections;
                    result.timestamp = pendingCtx.timestamp;
                    result.inferenceTime = std::chrono::duration<double, std::milli>(tInfer - pendingCtx.tStart).count();
                    
                    auto tEnd = std::chrono::high_resolution_clock::now();
                    double totalTime = std::chrono::duration<double, std::milli>(tEnd - pendingCtx.tStart).count();

                    cv::Mat rgb;
                    cv::cvtColor(pendingCtx.originalFrame, rgb, cv::COLOR_BGR2RGB);
                    QImage qimg((const uchar*)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
                    
                    onResultReady(result);
                    
                    emit frameReady(qimg.copy(), result);
                    emit fpsUpdate(1000.0 / totalTime);
                } catch (const dxrt::Exception& e) {
                    qCritical() << "YoloAsyncAdapter::processLoop() - DXRT ERROR in Wait (reqId=" << pendingCtx.reqId << "):" << e.what();
                    break;
                }
                hasPending = false;
            }

            // 2. Prepare and Submit Next Frame
            if (!m_source->read(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                continue;
            }
            if (frame.empty()) continue;

            RequestContext ctx;
            ctx.tStart = std::chrono::high_resolution_clock::now();
            ctx.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(ctx.tStart.time_since_epoch()).count();
            ctx.originalFrame = frame.clone();
            
            preprocess(frame, ctx.inputBuffer, ctx.scale, ctx.pad);
            
            try {
                ctx.reqId = m_engine->RunAsync(ctx.inputBuffer.data(), nullptr, nullptr);
                pendingCtx = std::move(ctx);
                hasPending = true;
            } catch (const dxrt::Exception& e) {
                qCritical() << "YoloAsyncAdapter::processLoop() - DXRT ERROR in RunAsync:" << e.what();
                break;
            }

        } catch (const dxrt::Exception& e) {
            qCritical() << "YoloAsyncAdapter::processLoop() - UNEXPECTED DXRT ERROR in loop:" << e.what();
            break;
        } catch (...) {
            qCritical() << "YoloAsyncAdapter::processLoop() - UNKNOWN ERROR in loop";
            break;
        }
    }
    if (m_source) m_source->close();
}

void YoloAsyncAdapter::onResultReady(DetectionResult& result) {
    // Default implementation does nothing
}
