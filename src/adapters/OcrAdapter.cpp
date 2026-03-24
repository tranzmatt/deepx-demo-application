#include "OcrAdapter.h"
#include <QJsonArray>
#include <QJsonObject>
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <QPainter>
#include <QFont>

using namespace std;
using namespace cv;

OcrAdapter::OcrAdapter(QObject* parent) : BaseAdapter(parent) {
    m_source = make_unique<VideoSource>();
    m_postProcess = make_unique<OcrPostProcess>();
}

OcrAdapter::~OcrAdapter() {
    stop();
}


bool OcrAdapter::loadConfig(const QJsonObject& config) {
    // 1. Detection Model
    QJsonObject detObj = config["det_model"].toObject();
    if (detObj.isEmpty()) {
        emit errorOccurred("Missing det_model config");
        return false;
    }
    std::string detPath = resolvePath(detObj["path"].toString());
    
    dxrt::InferenceOption io;
    try {
        qDebug() << "Loading Det Model:" << QString::fromStdString(detPath);
        m_detEngine = make_unique<dxrt::InferenceEngine>(detPath, io);
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
    m_detInputSize = detObj["input_size"].toInt(480); 
    
    // 2. Classification Model
    QJsonObject clsObj = config["cls_model"].toObject();
    if (!clsObj.isEmpty()) {
        std::string clsPath = resolvePath(clsObj["path"].toString());
        try {
            m_clsEngine = make_unique<dxrt::InferenceEngine>(clsPath, io);
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
    }
    
    // 3. Recognition Models
    QJsonObject recObj = config["rec_model"].toObject();
    QJsonArray recModels = recObj["models"].toArray();
    for (const auto& mConf : recModels) {
        QJsonObject m = mConf.toObject();
        int ratio = m["ratio"].toInt();
        int height = m["height"].toInt();
        std::string path = resolvePath(m["path"].toString());
        
        try {
            m_recEngines[ratio][height] = make_shared<dxrt::InferenceEngine>(path, io);
        } catch (...) {
            qWarning() << "Failed to load rec model:" << QString::fromStdString(path);
        }
    }
    
    // 4. Post Process Config
    QJsonObject ppObj = config["post_process"].toObject();
    m_postProcess->setDetThresh(ppObj["det_thresh"].toDouble(0.3));
    m_postProcess->setDetBoxThresh(ppObj["det_box_thresh"].toDouble(0.6));
    m_postProcess->setUnclipRatio(ppObj["unclip_ratio"].toDouble(1.5));
    
    QString charDictPath = ppObj["char_dict_path"].toString();
    if (!charDictPath.isEmpty()) {
        if (!m_postProcess->loadCharDict(resolvePath(charDictPath))) {
            qWarning() << "Failed to load char dict";
        }
    }

    // 5. Source
    QJsonObject inputObj = config["input"].toObject();
    QJsonArray sources = inputObj["sources"].toArray();
    if (sources.isEmpty()) {
        emit errorOccurred("No sources defined");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_imagePaths.clear();
        m_thumbnails.clear();
        for (const auto& s : sources) {
            QJsonObject srcObj = s.toObject();
            QString type = srcObj["type"].toString();
            QString path = srcObj["path"].toString();
            
            if (type == "camera" || type == "libcamera") {
                continue;
            }

            m_imagePaths.push_back(path);
            
            cv::Mat bgr = cv::imread(resolvePath(path));
            if (!bgr.empty()) {
                cv::Mat thumb;
                cv::resize(bgr, thumb, cv::Size(160, 120)); 
                cv::cvtColor(thumb, thumb, cv::COLOR_BGR2RGB);
                m_thumbnails.push_back(thumb);
            } else {
                m_thumbnails.push_back(cv::Mat(120, 160, CV_8UC3, cv::Scalar(50, 50, 50)));
            }
        }
        if (!m_imagePaths.empty()) {
            m_selectedImageIndex = 0;
            m_needsRecognition = true;
        }
    }

    m_sources.clear();
    for (const auto& s : sources) {
        QJsonObject srcObj = s.toObject();
        SourceInfo info;
        info.path = srcObj["path"].toString();
        info.type = srcObj["type"].toString();
        info.flip = srcObj["flip"].toBool(false);
        m_sources.push_back(info);
    }

    QJsonObject firstSourceObj = sources[0].toObject();
    if (firstSourceObj.contains("secondary_path") && m_sources.size() == 1) {
        SourceInfo secInfo;
        secInfo.path = firstSourceObj["secondary_path"].toString(m_sources[0].path);
        secInfo.type = firstSourceObj["secondary_type"].toString(m_sources[0].type);
        secInfo.flip = firstSourceObj["flip"].toBool(false);
        m_sources.push_back(secInfo);
    }

    if (!openSource(m_sources[0], m_source.get())) {
         emit errorOccurred(QString("Failed to open source: %1").arg(m_sources[0].path));
         return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_isStream = (m_sources[0].type == "camera" || m_sources[0].type == "video" || 
                      m_sources[0].type == "rtsp" || m_sources[0].type == "libcamera");
    }

    return true;
}

void OcrAdapter::handleMouseEvent(int x, int y, int type) {
    if (type != 0) return; // Only clicks

    std::lock_guard<std::mutex> lock(m_stateMutex);
    for (size_t i = 0; i < m_thumbnailRects.size(); ++i) {
        if (m_thumbnailRects[i].contains(x, y)) {
            if (m_selectedImageIndex != (int)i) {
                m_selectedImageIndex = i;
                m_needsRecognition = true;
                qDebug() << "Selected image index:" << i << m_imagePaths[i];
            }
            break;
        }
    }
}

void OcrAdapter::preprocessDet(const cv::Mat& img, cv::Mat& out, float& scale, std::vector<int>& pad) {
    int w = img.cols;
    int h = img.rows;
    scale = std::min((float)m_detInputSize / w, (float)m_detInputSize / h);
    int newW = (int)round(w * scale);
    int newH = (int)round(h * scale);
    
    // Resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(newW, newH));
    
    // Pad
    int dw = (m_detInputSize - newW) / 2;
    int dh = (m_detInputSize - newH) / 2;
    
    pad = {dw, dh};
    
    cv::copyMakeBorder(resized, out, dh, m_detInputSize - newH - dh, dw, m_detInputSize - newW - dw, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
}

cv::Mat OcrAdapter::getRotateCrop(const cv::Mat& img, const std::vector<cv::Point>& box) {
    std::vector<cv::Point2f> srcPts;
    for(auto p : box) srcPts.push_back(cv::Point2f(p));
    
    float w = std::max(cv::norm(srcPts[0]-srcPts[1]), cv::norm(srcPts[2]-srcPts[3]));
    float h = std::max(cv::norm(srcPts[0]-srcPts[3]), cv::norm(srcPts[1]-srcPts[2]));
    
    std::vector<cv::Point2f> dstPts = {
        {0, 0},
        {w, 0},
        {w, h},
        {0, h}
    };
    
    cv::Mat M = cv::getPerspectiveTransform(srcPts, dstPts);
    cv::Mat dst;
    cv::warpPerspective(img, dst, M, cv::Size((int)w, (int)h), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    
    if (dst.rows > dst.cols * 2) {
        cv::rotate(dst, dst, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    
    return dst;
}

std::shared_ptr<dxrt::InferenceEngine> OcrAdapter::getRecModel(int width, int height) {
    if (height == 0) height = 1;
    float ratio = (float)width / height;
    
    int ratioRes = 0;
    if (ratio < 10) ratioRes = 5;
    else if (ratio < 20) ratioRes = 15;
    else ratioRes = 25; 
    
    if (ratio >= 20) ratioRes = 25;
    else if (ratio >= 10) ratioRes = 15;
    else ratioRes = 5;
    
    int heightRes = 30;
    if (height < 12) heightRes = 10;
    else if (height < 24) heightRes = 20;
    
    if (m_recEngines.count(ratioRes) && m_recEngines[ratioRes].count(heightRes)) {
        return m_recEngines[ratioRes][heightRes];
    }
    
    if (!m_recEngines.empty()) {
        auto it = m_recEngines.begin();
        if (!it->second.empty()) return it->second.begin()->second;
    }
    
    return nullptr;
}

void OcrAdapter::processLoop() {
    cv::Mat frame;
    cv::Mat preprocessed;
    
    DetectionResult lastResult;
    cv::Mat currentOriginalFrame;
    QString lastPath;

    const int TOTAL_W = 1280;
    const int TOTAL_H = 720;
    const int TEXT_PANEL_W = 380;
    const int THUMB_PANEL_W = 200;
    const int IMAGE_AREA_W = TOTAL_W - TEXT_PANEL_W - THUMB_PANEL_W; // 700

    while (m_running) {
        if (m_reopenRequested) {
            m_reopenRequested = false;
            m_source->close();
            SourceInfo target = getCurrentSource();
            if (!openSource(target, m_source.get())) {
                 qWarning() << "[OCR] Failed to switch to source:" << target.path;
            } else {
                 std::lock_guard<std::mutex> lock(m_stateMutex);
                 m_isStream = (target.type == "camera" || target.type == "video" || 
                               target.type == "rtsp" || target.type == "libcamera");
                 m_needsRecognition = true;
            }
        }

        auto tStart = std::chrono::high_resolution_clock::now();
        
        bool isStream = false;
        int selectedIdx = -1;
        bool needsRec = false;
        {
            std::lock_guard<std::mutex> lock(m_stateMutex);
            selectedIdx = m_selectedImageIndex;
            needsRec = m_needsRecognition;
            isStream = m_isStream;
        }

        bool frameReadyToProcess = false;
        DetectionResult lastResult;

        if (isStream) {
            cv::Mat streamFrame;
            if (m_source->read(streamFrame) && !streamFrame.empty()) {
                currentOriginalFrame = streamFrame;
                cv::cvtColor(currentOriginalFrame, currentOriginalFrame, cv::COLOR_BGR2RGB);
                frameReadyToProcess = true;
                needsRec = true; // Always process new frames for streams
            } else {
                // Wait for next frame
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                continue;
            }
        } else if (selectedIdx >= 0 && selectedIdx < (int)m_imagePaths.size()) {
            QString path = m_imagePaths[selectedIdx];
            if (path != lastPath || needsRec) {
                lastPath = path;
                cv::Mat bgr = cv::imread(resolvePath(path));
                if (!bgr.empty()) {
                    cv::cvtColor(bgr, currentOriginalFrame, cv::COLOR_BGR2RGB);
                    frameReadyToProcess = true;
                }
            }
        }

        if (frameReadyToProcess && needsRec) {
            // Run OCR
                    lastResult.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(tStart.time_since_epoch()).count();
                    
                    cv::Mat rgbFrame = currentOriginalFrame;
                    
                    cv::Mat detImg;
                    cv::resize(rgbFrame, detImg, cv::Size(m_detInputSize, m_detInputSize));

                    auto detReq = m_detEngine->RunAsync(detImg.data, nullptr, nullptr);
                    auto detOut = m_detEngine->Wait(detReq);
                    
                    auto detShape = detOut[0]->shape();
                    int mapH = detShape.size() >= 4 ? detShape[2] : detShape.size() >= 2 ? detShape[0] : m_detInputSize;
                    int mapW = detShape.size() >= 4 ? detShape[3] : detShape.size() >= 2 ? detShape[1] : m_detInputSize;

                    std::vector<OcrBox> boxes = m_postProcess->postProcessDet((float*)detOut[0]->data(), mapW, mapH, m_detInputSize, m_detInputSize);
                    
                    for (const auto& ob : boxes) {
                        std::vector<cv::Point2f> realPoly;
                        std::vector<cv::Point> cropPoly;
                        int w = rgbFrame.cols;
                        int h = rgbFrame.rows;
                        for (const auto& pt : ob.box) {
                            float rx = (float)pt.x * w / m_detInputSize;
                            float ry = (float)pt.y * h / m_detInputSize;
                            realPoly.push_back(cv::Point2f(rx / w, ry / h));
                            cropPoly.push_back(cv::Point((int)rx, (int)ry));
                        }
                        
                        cv::Mat crop = getRotateCrop(rgbFrame, cropPoly); 
                        if (crop.empty()) continue;

                        // Check vertical rotation
                        if (crop.rows > crop.cols * 2) {
                            cv::rotate(crop, crop, cv::ROTATE_90_COUNTERCLOCKWISE);
                        }

                        if (m_clsEngine) {
                            cv::Mat clsIn;
                            cv::resize(crop, clsIn, cv::Size(192, 48));
                            auto clsReq = m_clsEngine->RunAsync(clsIn.data, nullptr, nullptr);
                            auto clsOut = m_clsEngine->Wait(clsReq);
                            auto clsRes = m_postProcess->postProcessCls((float*)clsOut[0]->data(), 2);
                            if (clsRes.first == 1 && clsRes.second > m_clsThresh) {
                                cv::rotate(crop, crop, cv::ROTATE_180);
                            }
                        }

                        auto enginePtr = getRecModel(crop.cols, crop.rows);
                        if (enginePtr) {
                            // Recognition Preprocessing: resize_norm_img_pad (Gray Padding)
                            float ratio = (float)crop.cols / crop.rows;
                            int targetH = 48;
                            int targetW = (ratio < 10) ? 240 : (ratio < 20) ? 720 : 1200;
                            
                            int resizedW = std::min(targetW, (int)std::ceil(targetH * ratio));
                            cv::Mat resized;
                            cv::resize(crop, resized, cv::Size(resizedW, targetH));
                            
                            cv::Mat recIn(targetH, targetW, CV_8UC3, cv::Scalar(127, 127, 127));
                            if (resizedW > 0) {
                                resized.copyTo(recIn(cv::Rect(0, 0, resizedW, targetH)));
                            }

                            auto recReq = enginePtr->RunAsync(recIn.data, nullptr, nullptr);
                            auto recOut = enginePtr->Wait(recReq);
                            if (!recOut.empty()) {
                                auto shape = recOut[0]->shape();
                                int actualSeqLen = (shape.size() >= 3) ? shape[1] : shape[0];
                                int actualNumClasses = (shape.size() >= 3) ? shape[2] : shape[1];
                                auto recRes = m_postProcess->postProcessRec((float*)recOut[0]->data(), actualSeqLen, actualNumClasses); 
                                if (recRes.second > 0.4) { 
                                    DetectionBox db;
                                    db.label = recRes.first;
                                    db.confidence = recRes.second;
                                    db.overlayPolygon = realPoly;
                                    db.box = cv::boundingRect(realPoly);
                                    lastResult.detections.push_back(db);
                                }
                            }
                        }
                    }
                    
                    std::lock_guard<std::mutex> lock(m_stateMutex);
                    // Draw on Original Frame and cache
                    m_lastAnnotated = drawOcrResults(currentOriginalFrame, lastResult);
                    m_lastTextPanel = createTextPanel(lastResult, TOTAL_H, TEXT_PANEL_W);
                    m_needsRecognition = false;
        }

        QImage finalFrame(TOTAL_W, TOTAL_H, QImage::Format_RGB888);
        finalFrame.fill(QColor(30, 30, 30));

        QPainter painter(&finalFrame);
        painter.setRenderHint(QPainter::Antialiasing);

        if (!m_lastAnnotated.isNull()) {
            float s = std::min((float)IMAGE_AREA_W / m_lastAnnotated.width(), (float)TOTAL_H / m_lastAnnotated.height());
            int sw = m_lastAnnotated.width() * s;
            int sh = m_lastAnnotated.height() * s;
            int sx = (IMAGE_AREA_W - sw) / 2;
            int sy = (TOTAL_H - sh) / 2;
            
            painter.drawImage(QRect(sx, sy, sw, sh), m_lastAnnotated);
        }

        if (!m_lastTextPanel.isNull()) {
            painter.drawImage(IMAGE_AREA_W, 0, m_lastTextPanel);
        }

        int thumbW = 160;
        int thumbH = 120;
        int margin = 20;

        {
            std::lock_guard<std::mutex> lock(m_stateMutex);
            m_thumbnailRects.clear();
            for (size_t i = 0; i < m_imagePaths.size(); ++i) {
                int ty = i * (thumbH + margin + 20) + margin; 
                if (ty + thumbH + 20 > TOTAL_H) break;
                
                QRect r(IMAGE_AREA_W + TEXT_PANEL_W + (THUMB_PANEL_W - thumbW) / 2, ty, thumbW, thumbH);
                m_thumbnailRects.push_back(r);
                
                // Draw actual thumbnail image
                if (i < m_thumbnails.size() && !m_thumbnails[i].empty()) {
                    QImage thumbQ((const uchar*)m_thumbnails[i].data, m_thumbnails[i].cols, m_thumbnails[i].rows, m_thumbnails[i].step, QImage::Format_RGB888);
                    painter.drawImage(r, thumbQ);
                } else {
                    painter.fillRect(r, QColor(50, 50, 50));
                }

                // Draw selection border
                QColor color = (selectedIdx == (int)i) ? QColor(0, 255, 0) : QColor(150, 150, 150);
                painter.setPen(QPen(color, 2));
                painter.drawRect(r);
                
                // Draw name
                QString name = QFileInfo(m_imagePaths[i]).fileName();
                if (name.length() > 20) name = name.left(17) + "...";
                painter.setPen(Qt::white);
                painter.setFont(QFont("Arial", 8));
                painter.drawText(QRect(r.x(), ty + thumbH + 2, r.width(), 18), Qt::AlignCenter, name);
            }
        }
        painter.end();

        auto tEnd = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        
        DetectionResult uiResult = lastResult;
        uiResult.detections.clear(); 
        
        emit frameReady(finalFrame.copy(), uiResult);
        emit fpsUpdate(-1.0);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(30)); 
    }
}

QImage OcrAdapter::drawOcrResults(const cv::Mat& img, const DetectionResult& result) {
    QImage drawImg((const uchar*)img.data, img.cols, img.rows, img.step, QImage::Format_RGB888);
    drawImg = drawImg.copy(); 

    QPainter painter(&drawImg);
    painter.setRenderHint(QPainter::Antialiasing);
    
    unsigned int seed = 0;
    auto getRandomColor = [&seed]() {
        seed = seed * 1664525 + 1013904223;
        return QColor(seed % 256, (seed / 256) % 256, (seed / 65536) % 256);
    };

    for (const auto& box : result.detections) {
        if (box.overlayPolygon.empty()) continue;
        
        QColor color = getRandomColor();
        QPolygon pts;
        for (const auto& p : box.overlayPolygon) {
            pts << QPoint(p.x * img.cols, p.y * img.rows);
        }
        
        painter.setPen(QPen(color, 2));
        painter.drawPolygon(pts);

        // Draw text label with UTF-8 support
        QString label = QString::fromStdString(box.label);
        QPoint textPos(pts[0].x(), std::max(pts[0].y() - 5, 20));
        
        QFont font("Arial", 10, QFont::Bold);
        painter.setFont(font);
        QFontMetrics fm(font);
        QRect textRect = fm.boundingRect(label).adjusted(0, 0, 4, 4);
        textRect.moveBottomLeft(textPos);

        painter.fillRect(textRect, color);
        painter.setPen(Qt::white);
        painter.drawText(textRect.adjusted(2, 0, 0, 0), Qt::AlignVCenter, label);
    }
    painter.end();
    return drawImg;
}

QImage OcrAdapter::createTextPanel(const DetectionResult& result, int h, int w) {
    QImage panel(w, h, QImage::Format_RGB888);
    panel.fill(Qt::white);

    QPainter painter(&panel);
    painter.setRenderHint(QPainter::TextAntialiasing);
    painter.setPen(Qt::black);
    
    // Draw right border
    painter.drawLine(w - 1, 0, w - 1, h - 1);

    QFont headerFont("Arial", 10, QFont::Bold);
    QFont itemFont("Arial", 9);
    int gap = 25;
    int y = gap;

    painter.setFont(headerFont);
    painter.drawText(5, y, "recognized text : confidence score");
    y += gap;

    painter.setFont(itemFont);
    int index = 1;
    for (const auto& box : result.detections) {
        QString label = QString::fromStdString(box.label);
        QString score = QString::number(box.confidence, 'f', 2);
        QString line = QString("%1. %2 : %3").arg(index++).arg(label).arg(score);
        
        painter.drawText(5, y, line);
        y += gap;
        if (y >= h - gap) break;
    }
    painter.end();
    return panel;
}
