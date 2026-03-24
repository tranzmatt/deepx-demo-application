#include "BaseAdapter.h"
#include "VideoSource.h"
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>
#include <sstream>

BaseAdapter::BaseAdapter(QObject* parent) : IAdapter(parent) {
    m_running = false;
}

BaseAdapter::~BaseAdapter() {
    stop();
}

void BaseAdapter::setSourceIndex(int index) {
    std::lock_guard<std::mutex> lock(m_sourceMutex);
    if (index >= 0 && index < (int)m_sources.size() && m_sourceIndex != index) {
        m_sourceIndex = index;
        m_reopenRequested = true;
        qDebug() << "[BaseAdapter] Source switch requested. Index:" << index;
    }
}

SourceInfo BaseAdapter::getCurrentSource() {
    std::lock_guard<std::mutex> lock(m_sourceMutex);
    if (m_sources.empty()) {
        return SourceInfo();
    }
    int idx = m_sourceIndex.load();
    if (idx < 0 || idx >= (int)m_sources.size()) idx = 0;
    return m_sources[idx];
}

std::string BaseAdapter::resolvePath(const QString& p) {
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
}

bool BaseAdapter::openSource(const SourceInfo& info, VideoSource* source) {
    if (!source) return false;
    
    bool isNumeric = false;
    info.path.toInt(&isNumeric);
    
    if (info.type == "libcamera") {
        // Parse path format: "index:width:height:fps"
        QStringList parts = info.path.split(':');
        int idx = 0, w = 1536, h = 864, fps = 30;
        if (parts.size() >= 1) idx = parts[0].toInt();
        if (parts.size() >= 2) w   = parts[1].toInt();
        if (parts.size() >= 3) h   = parts[2].toInt();
        if (parts.size() >= 4) fps = parts[3].toInt();
        qDebug() << "[BaseAdapter] Opening libcamera source idx:" << idx
                 << "resolution:" << w << "x" << h << "fps:" << fps;
        bool ok = source->openLibcamera(idx, w, h, fps);
        if (ok) source->setFlip(info.flip);
        return ok;
    } else if (info.type == "camera" && isNumeric) {
        bool ok = source->openCamera(info.path.toInt());
        if (ok) source->setFlip(info.flip);
        return ok;
    } else {
        std::string p = info.path.toStdString();
        // Don't resolve RTSP URLs or full /dev/ paths
        if (!p.empty() && p.find("://") == std::string::npos && p.rfind("/dev/", 0) != 0) {
            p = resolvePath(info.path);
        }
        bool ok = source->open(p);
        if (ok) source->setFlip(info.flip);
        return ok;
    }
}

void BaseAdapter::start() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_running) {
        return;
    }

    m_running = true;
    try {
        m_workerThread = std::thread(&BaseAdapter::processLoop, this);
    } catch (const std::exception& e) {
        qCritical() << "BaseAdapter::start() - failed to start thread:" << e.what();
        m_running = false;
    }
}

void BaseAdapter::stop() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_running) {
        return;
    }

    m_running = false;
    if (m_workerThread.joinable()) {
        if (m_workerThread.get_id() == std::this_thread::get_id()) {
            qCritical() << "BaseAdapter::stop() - ERROR: Thread is trying to join itself! This will cause deadlock.";
        } else {
            m_workerThread.join();
        }
    }
}
