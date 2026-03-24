#include "MultiChannelAdapter.h"
#include <QJsonArray>
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <chrono>

MultiChannelAdapter::MultiChannelAdapter(QObject* parent) : BaseAdapter(parent) {}

MultiChannelAdapter::~MultiChannelAdapter() {
    stop();
}

bool MultiChannelAdapter::loadConfig(const QJsonObject& config) {
    // Path resolver
    auto resolvePath = [](QString p) -> std::string {
        QFileInfo fi(p);
        if (fi.exists()) return p.toStdString();
        
        QDir rootDir(QCoreApplication::applicationDirPath());
        if (rootDir.dirName() == "build") rootDir.cdUp();
        
        QString absPath = rootDir.absoluteFilePath(p);
        if (QFileInfo::exists(absPath)) return absPath.toStdString();
        
        if (p.startsWith("./")) {
            absPath = rootDir.absoluteFilePath(p.mid(2));
            if (QFileInfo::exists(absPath)) return absPath.toStdString();
        }
        return p.toStdString();
    };

    // 1. Load Model (shared)
    QJsonObject modelObj = config["model"].toObject();
    QString modelPath = modelObj["path"].toString();
    std::string resolvedModelPath = resolvePath(modelPath);
    qDebug() << "[MultiChannel] Loading model from:" << QString::fromStdString(resolvedModelPath);

    dxrt::InferenceOption io;
    try {
        m_engine = std::make_shared<dxrt::InferenceEngine>(resolvedModelPath, io);
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
    
    // Detect model type
    // Check explicit decoding method for Face
    if (modelParam["decoding_method"].toString() == "yolo_face") {
        m_modelType = YOLO_FACE;
        m_postProcessFace = std::make_unique<YoloFacePostProcess>(m_inputWidth, m_inputHeight, scoreThresh, iouThresh);
        qDebug() << "[MultiChannel] Model Type: YOLO Face (Explicit 'yolo_face')";
    }
    // Check for Pose
    else if (resolvedModelPath.find("pose") != std::string::npos) {
        m_modelType = YOLO_POSE;
        m_postProcessPose = std::make_unique<YoloPosePostProcess>(m_inputWidth, m_inputHeight, scoreThresh, iouThresh);
        qDebug() << "[MultiChannel] Model Type: YOLO Pose";
    }
    // Check V8 vs standard by anchors
    else {
        QJsonArray layers = modelParam["layer"].toArray();
        bool hasAnchors = false;
        for(const auto& val: layers) {
             if(!val.toObject().value("anchor_width").toArray().isEmpty()) {
                 hasAnchors = true; 
                 break;
             }
        }

        if (!hasAnchors) {
            m_modelType = YOLO_V8_PPU;
            m_postProcessV8 = std::make_unique<YoloV8PpuPostProcess>(m_inputWidth, m_inputHeight, scoreThresh, iouThresh);
            qDebug() << "[MultiChannel] Model Type: YOLOv8 PPU";
        } else {
            m_modelType = YOLO_STANDARD;
            m_postProcess = std::make_unique<YoloPostProcess>(m_inputWidth, m_inputHeight, 0.25f, scoreThresh, iouThresh);
            qDebug() << "[MultiChannel] Model Type: Standard YOLO";
        }
    }
    
    // Load class names
    QJsonObject outputObj = config["output"].toObject();
    QJsonArray classes = outputObj["classes"].toArray();
    m_classNames.clear();
    for (const auto& c : classes) {
        m_classNames.push_back(c.toString().toStdString());
    }

    // 3. Open all sources
    QJsonObject inputObj = config["input"].toObject();
    QJsonArray sources = inputObj["sources"].toArray();
    
    int numChannels = sources.size();
    if (numChannels == 0) {
        emit errorOccurred("No sources defined");
        return false;
    }
    
    // Grid layout calculation
    m_gridDim = (int)std::ceil(std::sqrt((double)numChannels));
    m_compositeWidth = 1920;  // Fixed composite size
    m_compositeHeight = 1080;
    int cellWidth = m_compositeWidth / m_gridDim;
    int cellHeight = m_compositeHeight / m_gridDim;
    
    qDebug() << "[MultiChannel] Channels:" << numChannels << "Grid:" << m_gridDim << "x" << m_gridDim;
    
    m_compositeFrame = cv::Mat(m_compositeHeight, m_compositeWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    
    for (int i = 0; i < numChannels; ++i) {
        QJsonObject srcObj = sources[i].toObject();
        QString srcPath = srcObj["path"].toString();
        QString type = srcObj["type"].toString();
        
        auto ctx = std::make_unique<ChannelContext>();
        ctx->index = i;
        ctx->source = std::make_unique<VideoSource>();
        
        // Grid position
        ctx->posX = (i % m_gridDim) * cellWidth;
        ctx->posY = (i / m_gridDim) * cellHeight;
        ctx->width = cellWidth;
        ctx->height = cellHeight;
        ctx->resultFrame = cv::Mat(cellHeight, cellWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        
        ctx->primaryPath = srcPath;
        ctx->primaryType = type;
        ctx->secondaryPath = srcObj["secondary_path"].toString(srcPath);
        ctx->secondaryType = srcObj["secondary_type"].toString(type);

        if (!openSource({srcPath, type}, ctx->source.get())) {
            qWarning() << "[MultiChannel] Failed to open source" << i << ":" << srcPath;
            // Draw "No Signal" placeholder
            cv::putText(ctx->resultFrame, "No Signal", cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        }
        
        m_channels.push_back(std::move(ctx));
    }
    
    // 4. Load Trespassing Points
    if (config.contains("points")) {
        m_trespassingPolygon.clear();
        QJsonArray pointsArr = config["points"].toArray();
        QJsonArray resArr = config["application"].toObject()["resolution"].toArray();
        float refW = resArr[0].toInt(960); // Default ref width
        float refH = resArr[1].toInt(540); // Default ref height
        
        qDebug() << "[MultiChannel] Loading Trespassing Polygon (" << pointsArr.size() << " points)";
        
        for (const auto& pVal : pointsArr) {
            QJsonArray pt = pVal.toArray();
            float x = pt[0].toInt();
            float y = pt[1].toInt();
            m_trespassingPolygon.push_back({x / refW, y / refH});
        }
    }
    
    return true;
}

void MultiChannelAdapter::start() {
    m_running = true;
    
    // Start per-channel threads
    for (auto& ctx : m_channels) {
        ctx->running = true;
        ctx->thread = std::thread(&MultiChannelAdapter::channelLoop, this, ctx->index);
    }
    
    // Start composite emitter thread
    m_workerThread = std::thread(&MultiChannelAdapter::processLoop, this);
}

void MultiChannelAdapter::stop() {
    m_running = false;
    
    // Stop all channel threads
    for (auto& ctx : m_channels) {
        ctx->running = false;
        if (ctx->thread.joinable()) {
            ctx->thread.join();
        }
        // Explicitly close source after thread joins but before object destruction
        if (ctx->source) {
            ctx->source->close();
            // Stagger closures slightly to help GStreamer settle
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    
    // Stop composite thread
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
}


void MultiChannelAdapter::preprocess(const cv::Mat& img, std::vector<uint8_t>& buffer, float& scale, std::vector<int>& pad) {
    int w = img.cols, h = img.rows;
    scale = std::min((float)m_inputWidth / w, (float)m_inputHeight / h);
    int nw = (int)(w * scale), nh = (int)(h * scale);
    
    int padW = (m_inputWidth - nw) / 2;
    int padH = (m_inputHeight - nh) / 2;
    pad = {padW, padH, m_inputWidth - nw - padW, m_inputHeight - nh - padH};
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh));
    
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, padH, pad[3], padW, pad[2], 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    buffer.assign(padded.data, padded.data + padded.total() * padded.elemSize());
}

void MultiChannelAdapter::channelLoop(int channelIndex) {
    auto& ctx = m_channels[channelIndex];
    
    while (ctx->running && m_running) {
        if (ctx->reopenRequested) {
            ctx->reopenRequested = false;
            ctx->source->close();
            
            QString targetPath = ctx->useSecondary ? ctx->secondaryPath : ctx->primaryPath;
            QString targetType = ctx->useSecondary ? ctx->secondaryType : ctx->primaryType;
            
            if (!openSource({targetPath, targetType}, ctx->source.get())) {
                qWarning() << "[MultiChannel] Channel" << channelIndex << "failed to switch to" << targetPath;
            } else {
                qDebug() << "[MultiChannel] Channel" << channelIndex << "switched to" << targetPath;
            }
        }

        cv::Mat frame;
        if (!ctx->source->read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }
        
        auto tStart = std::chrono::high_resolution_clock::now();
        
        // Preprocess
        std::vector<uint8_t> inputBuffer;
        float scale;
        std::vector<int> pad;
        preprocess(frame, inputBuffer, scale, pad);
        
        // Inference (shared engine)
        dxrt::TensorPtrs outputs;
        try {
            auto reqId = m_engine->RunAsync(inputBuffer.data());
            outputs = m_engine->Wait(reqId);
        } catch (const dxrt::Exception& e) {
            qWarning() << "[MultiChannel] Inference error (DXRT) on channel" << channelIndex << ":" << e.what();
            continue;
        } catch (const std::exception& e) {
            qWarning() << "[MultiChannel] Inference error on channel" << channelIndex << ":" << e.what();
            continue;
        } catch (...) {
            qWarning() << "[MultiChannel] Inference error (unknown) on channel" << channelIndex;
            continue;
        }
        
        // Post-process
        DetectionResult result;
        
        std::vector<DetectionBox> detections;
        if (m_modelType == YOLO_V8_PPU) {
            detections = m_postProcessV8->postprocess(outputs, m_inputWidth, m_inputHeight);
        } else if (m_modelType == YOLO_FACE) {
            detections = m_postProcessFace->postprocess(outputs, m_inputWidth, m_inputHeight);
        } else if (m_modelType == YOLO_POSE) {
            detections = m_postProcessPose->postprocess(outputs, m_inputWidth, m_inputHeight);
        } else {
            detections = m_postProcess->postprocess(outputs, m_inputWidth, m_inputHeight);
        }
        
        // Adjust Coords (from model input to original frame)
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
            
            if (box.classId < m_classNames.size()) {
                box.label = m_classNames[box.classId];
            } else {
                box.label = "Unknown";
            }
        }
        result.detections = detections;
        
        auto tEnd = std::chrono::high_resolution_clock::now();
        result.inferenceTime = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        
        // Calculate Processing FPS
        double totalTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        if (totalTimeMs > 0) {
            double currentFps = 1000.0 / totalTimeMs;
            // Exponential moving average for smoothing
            ctx->fps = 0.9 * ctx->fps + 0.1 * currentFps;
        }
        
        // Resize frame to cell size
        cv::Mat cellFrame;
        cv::resize(frame, cellFrame, cv::Size(ctx->width, ctx->height));
        
        // Check Trespassing
        bool trespassingDetected = false;
        std::vector<cv::Point> polyPts;
        if (!m_trespassingPolygon.empty()) {
            for(auto& p : m_trespassingPolygon) {
                polyPts.push_back(cv::Point(p.x * cellFrame.cols, p.y * cellFrame.rows));
            }
            
            for (const auto& det : result.detections) {
                if (det.label == "person") {
                    cv::Point2f feet;
                    feet.x = det.box.x + det.box.width / 2;
                    feet.y = det.box.y + det.box.height; 
                    
                    if (isPointInPolygon(feet, m_trespassingPolygon)) {
                        trespassingDetected = true;
                    }
                }
            }
        }

        // Draw Polygon
        if (!polyPts.empty()) {
            const cv::Point* ppt[1] = { &polyPts[0] };
            int npt[] = { (int)polyPts.size() };
            cv::Scalar polyColor = trespassingDetected ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::polylines(cellFrame, ppt, npt, 1, true, polyColor, 2, cv::LINE_AA);
            
            if (trespassingDetected) {
                cv::putText(cellFrame, "TRESPASSING!", cv::Point(20, 60), 
                       cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }
        }

        // Draw detections on cell
        for (const auto& det : result.detections) {
            int dx = det.box.x * cellFrame.cols;
            int dy = det.box.y * cellFrame.rows;
            int dw = det.box.width * cellFrame.cols;
            int dh = det.box.height * cellFrame.rows;
            
            cv::Scalar color(det.color[0], det.color[1], det.color[2]);
            
            // Highlight trespassing object
            cv::Point2f feet;
            feet.x = det.box.x + det.box.width / 2;
            feet.y = det.box.y + det.box.height;
            if (!m_trespassingPolygon.empty() && isPointInPolygon(feet, m_trespassingPolygon)) {
                color = cv::Scalar(0, 0, 255); // Red for intruder
            }

            cv::rectangle(cellFrame, cv::Rect(dx, dy, dw, dh), color, 2);
            
            std::string label = det.label + " " + std::to_string((int)(det.confidence * 100)) + "%";
            
            int baseLine;
            double fontScale = 0.5;
            int thickness = 1;
            int fontFace = cv::FONT_HERSHEY_DUPLEX;
            
            cv::Size labelSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseLine);
            int textY = std::max(dy, labelSize.height + 5);
            
            cv::rectangle(cellFrame, cv::Point(dx, textY - labelSize.height - 5),
                          cv::Point(dx + labelSize.width, textY + baseLine), color, cv::FILLED);
                          
            cv::putText(cellFrame, label, cv::Point(dx, textY), 
                       fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        }
        
        // Draw channel number
        cv::rectangle(cellFrame, cv::Point(0, 0), cv::Point(30, 20), cv::Scalar(0, 0, 255), cv::FILLED);
        cv::putText(cellFrame, std::to_string(channelIndex + 1), cv::Point(5, 15), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // Draw FPS
        QString fpsText = QString("FPS: %1").arg(ctx->fps, 0, 'f', 1);
        cv::putText(cellFrame, fpsText.toStdString(), cv::Point(cellFrame.cols - 70, 15), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        
        // Update channel result frame
        {
            std::lock_guard<std::mutex> lock(ctx->frameMutex);
            ctx->resultFrame = cellFrame;
            ctx->detections = result.detections;
        }
    }
}

void MultiChannelAdapter::processLoop() {
    m_lastGlobalTime = std::chrono::high_resolution_clock::now();
    while (m_running) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(now - m_lastGlobalTime).count();
        m_lastGlobalTime = now;
        
        if (elapsedMs > 0) {
            emit fpsUpdate(1000.0 / elapsedMs);
        }

        // Compose all channel frames into composite
        {
            std::lock_guard<std::mutex> lock(m_compositeMutex);
            for (auto& ctx : m_channels) {
                std::lock_guard<std::mutex> chLock(ctx->frameMutex);
                cv::Rect roi(ctx->posX, ctx->posY, ctx->width, ctx->height);
                ctx->resultFrame.copyTo(m_compositeFrame(roi));
            }
        }
        
        // Convert to QImage and emit
        cv::Mat rgb;
        cv::cvtColor(m_compositeFrame, rgb, cv::COLOR_BGR2RGB);
        QImage qimg((const uchar*)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        
        DetectionResult compositeResult;
        compositeResult.inferenceTime = 0; // Could aggregate
        
        emit frameReady(qimg.copy(), compositeResult);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps composite refresh
    }
}

bool MultiChannelAdapter::isPointInPolygon(const cv::Point2f& pt, const std::vector<cv::Point2f>& polygon) {
    bool inside = false;
    for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
        if (((polygon[i].y > pt.y) != (polygon[j].y > pt.y)) &&
            (pt.x < (polygon[j].x - polygon[i].x) * (pt.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x)) {
            inside = !inside;
        }
    }
    return inside;
}
