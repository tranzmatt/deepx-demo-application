#include "FaceAlignmentAdapter.h"
#include <QJsonArray>
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <vector>

static std::vector<float> softmax(const float* input, int size) {
    std::vector<float> output(size);
    float maxVal = input[0];
    for (int i = 1; i < size; i++) if (input[i] > maxVal) maxVal = input[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = std::exp(input[i] - maxVal);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) output[i] /= sum;
    return output;
}

static std::vector<float> sigmoid(const float* input, int size) {
    std::vector<float> output(size);
    for (int i = 0; i < size; i++) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
    return output;
}

static cv::Scalar getEmotionColor(const std::string& label) {
    std::string lowerLabel = label;
    std::transform(lowerLabel.begin(), lowerLabel.end(), lowerLabel.begin(), ::tolower);
    
    if (lowerLabel == "anger" || lowerLabel == "angry") return cv::Scalar(0, 0, 255);      // Red
    if (lowerLabel == "contempt") return cv::Scalar(0, 128, 255);                            // Orange-ish
    if (lowerLabel == "disgust") return cv::Scalar(0, 255, 255);                             // Yellow
    if (lowerLabel == "fear") return cv::Scalar(255, 0, 255);                                // Magenta
    if (lowerLabel == "happy") return cv::Scalar(0, 255, 0);                                 // Green
    if (lowerLabel == "neutral") return cv::Scalar(255, 255, 0);                             // Cyan
    if (lowerLabel == "sad") return cv::Scalar(255, 128, 0);                                 // Orange
    if (lowerLabel == "surprise" || lowerLabel == "suprised") return cv::Scalar(255, 0, 0);  // Blue
    
    return cv::Scalar(0, 255, 0); // Default Green
}

FaceAlignmentAdapter::FaceAlignmentAdapter(QObject* parent) : BaseAdapter(parent) {
    m_source = std::make_unique<VideoSource>();
}

FaceAlignmentAdapter::~FaceAlignmentAdapter() {
    stop();
}

bool FaceAlignmentAdapter::loadConfig(const QJsonObject& config) {
    QJsonObject modelObj = config["model"].toObject();
    QString modelPath = modelObj["path"].toString();

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

    QJsonObject modelParam = modelObj["param"].toObject();
    m_inputWidth = modelParam["input_width"].toInt(640);
    m_inputHeight = modelParam["input_height"].toInt(640);
    float scoreThresh = modelParam["score_threshold"].toDouble(0.25);
    float iouThresh = modelParam["iou_threshold"].toDouble(0.45);

    m_postProcess = std::make_unique<YoloFacePostProcess>(m_inputWidth, m_inputHeight, scoreThresh, iouThresh);

    // Load emotion model
    QJsonObject emotionModelObj = config["emotion_model"].toObject();
    if (!emotionModelObj.isEmpty()) {
        QString emotionModelPath = emotionModelObj["path"].toString();
        std::string resolvedEmotionPath = resolvePath(emotionModelPath);
        QJsonObject emotionParam = emotionModelObj["param"].toObject();
        m_emotionInputWidth = emotionParam["input_width"].toInt(224);
        m_emotionInputHeight = emotionParam["input_height"].toInt(224);

        dxrt::InferenceOption emotionIo;
        try {
            m_emotionEngine = std::make_unique<dxrt::InferenceEngine>(resolvedEmotionPath, emotionIo);
            qDebug() << "[Emotion] Model loaded successfully, input:" << m_emotionInputWidth << "x" << m_emotionInputHeight;
            
            QJsonArray classesArr = emotionParam["classes"].toArray();
            m_emotionClasses.clear();
            for (const auto& cls : classesArr) {
                m_emotionClasses.push_back(cls.toString().toStdString());
            }
            if (m_emotionClasses.empty()) {
                m_emotionClasses = {"neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"};
            }
        } catch (const dxrt::Exception& e) {
            qDebug() << "[Emotion] WARNING: Failed to load emotion model:" << e.what();
        } catch (const std::exception& e) {
            qDebug() << "[Emotion] WARNING: Failed to load emotion model:" << e.what();
        } catch (...) {
            qDebug() << "[Emotion] WARNING: Failed to load emotion model: unhandled exception";
        }
    }

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

    qDebug() << "[FaceDetect] Source opened successfully";
    return true;
}

void FaceAlignmentAdapter::preprocess(const cv::Mat& img, cv::Mat& preprocessed, float& scale, std::vector<int>& pad) {
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

// Simple NMS for decoded boxes
struct RawFace {
    float cx, cy, w, h, conf;
    float landmarks[10]; // 5 x,y pairs
    float area() const { return w * h; }
    float iou(const RawFace& o) const {
        float ax1 = cx - w/2, ay1 = cy - h/2, ax2 = cx + w/2, ay2 = cy + h/2;
        float bx1 = o.cx - o.w/2, by1 = o.cy - o.h/2, bx2 = o.cx + o.w/2, by2 = o.cy + o.h/2;
        float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
        float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
        float inter = std::max(0.f, ix2-ix1) * std::max(0.f, iy2-iy1);
        return inter / (area() + o.area() - inter + 1e-6f);
    }
};

void FaceAlignmentAdapter::processLoop() {
    cv::Mat frame, preprocessed;

    int frameCount = 0;
    while (m_running) {
        if (m_reopenRequested) {
            m_reopenRequested = false;
            m_source->close();
            SourceInfo target = getCurrentSource();
            if (!openSource(target, m_source.get())) {
                 qCritical() << "[FaceDetect] Failed to switch to source:" << target.path;
            }
        }

        auto tStart = std::chrono::high_resolution_clock::now();
        if (!m_source->read(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }
        if (frame.empty()) continue;
        frameCount++;

        float scale;
        std::vector<int> pad;
        preprocess(frame, preprocessed, scale, pad);

        std::vector<std::shared_ptr<dxrt::Tensor>> outputs;
        try {
            auto reqId = m_engine->RunAsync(preprocessed.data, nullptr, nullptr);
            outputs = m_engine->Wait(reqId);
        } catch (const dxrt::Exception& e) {
            qCritical() << "[FaceDetect] Inference EXCEPTION DXRT:" << e.what();
            continue;
        } catch (const std::exception& e) {
            qCritical() << "[FaceDetect] Inference EXCEPTION:" << e.what();
            continue;
        } catch (...) {
            qCritical() << "[FaceDetect] Inference UNKNOWN EXCEPTION";
            continue;
        }

        auto tInfer = std::chrono::high_resolution_clock::now();

        std::vector<RawFace> candidates;
        float confThreshold = 0.3f;

        if (!outputs.empty()) {
            auto& tensor = outputs[0];
            auto shape = tensor->shape();
            const float* data = static_cast<const float*>(tensor->data());

            int numBoxes = (shape.size() >= 2) ? shape[1] : 0;
            int elemPerBox = (shape.size() >= 3) ? shape[2] : 0;

            for (int i = 0; i < numBoxes; i++) {
                const float* det = data + i * elemPerBox;

                float objConf = det[4];

                if (objConf < confThreshold) continue;

                // Class confidence
                float clsConf = (elemPerBox > 5) ? det[5] : 1.0f;
                float finalConf = objConf * clsConf;

                if (finalConf < confThreshold) continue;

                RawFace face;
                face.cx = det[0];
                face.cy = det[1];
                face.w = det[2];
                face.h = det[3];
                face.conf = finalConf;

                // Landmarks (indices 6-15)
                for (int k = 0; k < 10 && (6 + k) < elemPerBox; k++) {
                    face.landmarks[k] = det[6 + k];
                }

                candidates.push_back(face);
            }
        }

        // NMS
        std::sort(candidates.begin(), candidates.end(), [](const RawFace& a, const RawFace& b) {
            return a.conf > b.conf;
        });
        std::vector<RawFace> nmsResults;
        std::vector<bool> suppressed(candidates.size(), false);
        for (size_t i = 0; i < candidates.size(); i++) {
            if (suppressed[i]) continue;
            nmsResults.push_back(candidates[i]);
            for (size_t j = i + 1; j < candidates.size(); j++) {
                if (!suppressed[j] && candidates[i].iou(candidates[j]) > 0.45f) {
                    suppressed[j] = true;
                }
            }
        }

        // Draw detections on frame
        cv::Mat display = frame.clone();
        std::vector<DetectionBox> detections;

        for (const auto& face : nmsResults) {
            // Convert from input pixel coords to original frame coords
            float x1 = (face.cx - face.w / 2.0f - pad[0]) / scale;
            float y1 = (face.cy - face.h / 2.0f - pad[1]) / scale;
            float bw = face.w / scale;
            float bh = face.h / scale;

            // Clamp to frame
            x1 = std::max(0.0f, x1);
            y1 = std::max(0.0f, y1);
            bw = std::min(bw, (float)frame.cols - x1);
            bh = std::min(bh, (float)frame.rows - y1);

            // ============================================================
            // Emotion recognition on cropped face
            // ============================================================
            std::string emotionLabel = "";
            float bestVal = 0.0f;
            int bestIdx = -1;
            std::vector<float> emotionProbs;
            cv::Scalar boxColor(0, 255, 0); // Default to green if no emotion engine

            if (m_emotionEngine && bw > 10 && bh > 10) {
                // Crop face from original frame
                cv::Rect faceROI((int)x1, (int)y1, (int)bw, (int)bh);
                cv::Mat faceCrop = frame(faceROI).clone();

                // Resize to emotion model input size (224x224)
                cv::Mat emotionInput;
                cv::resize(faceCrop, emotionInput, cv::Size(m_emotionInputWidth, m_emotionInputHeight));
                cv::cvtColor(emotionInput, emotionInput, cv::COLOR_BGR2RGB);

                try {
                    auto emotionReqId = m_emotionEngine->RunAsync(emotionInput.data, nullptr, nullptr);
                    auto emotionOutputs = m_emotionEngine->Wait(emotionReqId);

                    if (!emotionOutputs.empty()) {
                        auto& tensor = emotionOutputs[0];
                        auto shape = tensor->shape();
                        int numClasses = 1;
                        for (auto s : shape) numClasses *= s;

                        const float* edata = static_cast<const float*>(tensor->data());
                        emotionProbs = softmax(edata, numClasses);

                        // Find argmax
                        bestIdx = 0;
                        bestVal = emotionProbs[0];
                        for (int v = 1; v < numClasses; v++) {
                            if (emotionProbs[v] > bestVal) {
                                bestVal = emotionProbs[v];
                                bestIdx = v;
                            }
                        }

                        if (bestIdx < (int)m_emotionClasses.size()) {
                            std::string labelName = m_emotionClasses[bestIdx];
                            boxColor = getEmotionColor(labelName);
                            emotionLabel = labelName + " " + std::to_string((int)(bestVal * 100)) + "%";
                        }
                    }
                } catch (const dxrt::Exception& e) {
                    qCritical() << "[Emotion] Inference error DXRT:" << e.what();
                } catch (const std::exception& e) {
                    qCritical() << "[Emotion] Inference error:" << e.what();
                } catch (...) {
                    qCritical() << "[Emotion] Inference error: unhandled exception";
                }
            }

            // Build DetectionBox for result
            DetectionBox dbox;
            dbox.classId = bestIdx >= 0 ? bestIdx : 0;
            dbox.label = !emotionLabel.empty() ? m_emotionClasses[bestIdx] : "";
            dbox.confidence = bestIdx >= 0 ? bestVal : face.conf;
            dbox.box.x = x1 / frame.cols;
            dbox.box.y = y1 / frame.rows;
            dbox.box.width = bw / frame.cols;
            dbox.box.height = bh / frame.rows;
            dbox.color = boxColor;
            detections.push_back(dbox);
        }

        DetectionResult result;
        result.detections = detections;
        result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(tStart.time_since_epoch()).count();
        result.inferenceTime = std::chrono::duration<double, std::milli>(tInfer - tStart).count();

        auto tEnd = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        cv::Mat rgb;
        cv::cvtColor(display, rgb, cv::COLOR_BGR2RGB);
        QImage qimg((const uchar*)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        emit frameReady(qimg.copy(), result);
        emit fpsUpdate(1000.0 / totalTime);
    }
}
