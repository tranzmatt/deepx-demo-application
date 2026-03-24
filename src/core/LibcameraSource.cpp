#include "LibcameraSource.h"

#ifdef HAVE_LIBCAMERA

#include <libcamera/libcamera.h>
#include <sys/mman.h>
#include <unistd.h>
#include <QDebug>
#include <cstring>

using namespace libcamera;

// ---------- Construction / Destruction ----------

LibcameraSource::LibcameraSource() = default;

LibcameraSource::~LibcameraSource() {
    close();
}

// ---------- Open Camera ----------

bool LibcameraSource::open(int cameraIndex, int width, int height, double fps) {
    // Clean up any previous session
    close();

    // --- 1. CameraManager ---
    m_cameraManager = std::make_unique<CameraManager>();
    int ret = m_cameraManager->start();
    if (ret < 0) {
        qWarning() << "[LibcameraSource] Failed to start CameraManager:" << ret;
        close();
        return false;
    }

    auto cameras = m_cameraManager->cameras();
    if (cameras.empty()) {
        qWarning() << "[LibcameraSource] No cameras detected by libcamera";
        close();
        return false;
    }

    if (cameraIndex < 0 || cameraIndex >= static_cast<int>(cameras.size())) {
        qWarning() << "[LibcameraSource] Invalid camera index" << cameraIndex
                    << "(available:" << cameras.size() << ")";
        close();
        return false;
    }

    m_camera = cameras[cameraIndex];
    ret = m_camera->acquire();
    if (ret < 0) {
        qWarning() << "[LibcameraSource] Failed to acquire camera:" << ret;
        close();
        return false;
    }

    qInfo() << "[LibcameraSource] Camera acquired:" << m_camera->id().c_str();

    // --- 2. Configuration ---
    m_config = m_camera->generateConfiguration({StreamRole::VideoRecording});
    if (!m_config || m_config->empty()) {
        qWarning() << "[LibcameraSource] Failed to generate camera configuration";
        close();
        return false;
    }

    StreamConfiguration& streamCfg = m_config->at(0);

    // Try BGR/RGB first, fall back to NV12/YUV
    static const std::vector<std::pair<PixelFormat, PixFmt>> fmtPriority = {
        {formats::BGR888,  PixFmt::BGR888},
        {formats::RGB888,  PixFmt::RGB888},
        {formats::NV12,    PixFmt::NV12},
        {formats::NV21,    PixFmt::NV21},
        {formats::YUV420,  PixFmt::YUV420},
        {formats::YUYV,    PixFmt::YUYV},
    };

    bool fmtSet = false;
    for (auto& [pf, pft] : fmtPriority) {
        streamCfg.pixelFormat = pf;
        streamCfg.size = Size(width, height);
        auto status = m_config->validate();
        if (status == CameraConfiguration::Valid || status == CameraConfiguration::Adjusted) {
            m_pixFmt = pft;
            fmtSet = true;
            break;
        }
    }
    if (!fmtSet) {
        qWarning() << "[LibcameraSource] Could not find a suitable pixel format";
        close();
        return false;
    }

    ret = m_camera->configure(m_config.get());
    if (ret < 0) {
        qWarning() << "[LibcameraSource] Failed to configure camera:" << ret;
        close();
        return false;
    }

    m_stream = streamCfg.stream();
    m_width  = streamCfg.size.width;
    m_height = streamCfg.size.height;
    m_stride = streamCfg.stride;
    m_fps    = fps;

    qInfo() << "[LibcameraSource] Configured:"
            << m_width << "x" << m_height
            << "stride:" << m_stride
            << "fmt:" << streamCfg.pixelFormat.toString().c_str();

    // --- 3. Allocate buffers ---
    m_allocator = std::make_unique<FrameBufferAllocator>(m_camera);
    ret = m_allocator->allocate(m_stream);
    if (ret < 0) {
        qWarning() << "[LibcameraSource] Failed to allocate buffers:" << ret;
        close();
        return false;
    }

    const std::vector<std::unique_ptr<FrameBuffer>>& buffers = m_allocator->buffers(m_stream);
    qInfo() << "[LibcameraSource] Allocated" << buffers.size() << "buffers";

    // Memory-map each buffer's first plane
    for (const auto& buf : buffers) {
        const FrameBuffer::Plane& plane = buf->planes()[0];
        void* mapped = mmap(nullptr, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), plane.offset);
        if (mapped == MAP_FAILED) {
            qWarning() << "[LibcameraSource] mmap failed for buffer";
            close();
            return false;
        }
        m_mappedBuffers[buf.get()] = {mapped, plane.length};
    }

    // --- 4. Create capture requests ---
    for (const auto& buf : buffers) {
        auto request = m_camera->createRequest();
        if (!request) {
            qWarning() << "[LibcameraSource] Failed to create capture request";
            close();
            return false;
        }
        ret = request->addBuffer(m_stream, buf.get());
        if (ret < 0) {
            qWarning() << "[LibcameraSource] Failed to add buffer to request:" << ret;
            close();
            return false;
        }
        m_requests.push_back(std::move(request));
    }

    // --- 5. Connect signal & start ---
    m_camera->requestCompleted.connect(this, &LibcameraSource::requestComplete);

    ret = m_camera->start();
    if (ret < 0) {
        qWarning() << "[LibcameraSource] Failed to start camera:" << ret;
        close();
        return false;
    }
    m_active = true;

    // Queue all requests
    for (auto& req : m_requests) {
        m_camera->queueRequest(req.get());
    }

    qInfo() << "[LibcameraSource] Camera started successfully";
    return true;
}

// ---------- Request Complete Callback ----------

void LibcameraSource::requestComplete(Request* request) {
    if (request->status() == Request::RequestCancelled) return;

    const auto& buffers = request->buffers();
    auto it = buffers.find(m_stream);
    if (it == buffers.end()) {
        // Re-queue and return
        request->reuse(Request::ReuseBuffers);
        m_camera->queueRequest(request);
        return;
    }

    const FrameBuffer* fb = it->second;
    auto mit = m_mappedBuffers.find(fb);
    if (mit == m_mappedBuffers.end()) {
        request->reuse(Request::ReuseBuffers);
        m_camera->queueRequest(request);
        return;
    }

    void* data = mit->second.data;
    cv::Mat frame;

    switch (m_pixFmt) {
        case PixFmt::BGR888: {
            cv::Mat raw(m_height, m_width, CV_8UC3, data, m_stride);
            frame = raw.clone();
            break;
        }
        case PixFmt::RGB888: {
            cv::Mat raw(m_height, m_width, CV_8UC3, data, m_stride);
            cv::cvtColor(raw, frame, cv::COLOR_RGB2BGR);
            break;
        }
        case PixFmt::NV12: {
            cv::Mat yuv(m_height * 3 / 2, m_width, CV_8UC1, data, m_stride);
            cv::cvtColor(yuv, frame, cv::COLOR_YUV2BGR_NV12);
            break;
        }
        case PixFmt::NV21: {
            cv::Mat yuv(m_height * 3 / 2, m_width, CV_8UC1, data, m_stride);
            cv::cvtColor(yuv, frame, cv::COLOR_YUV2BGR_NV21);
            break;
        }
        case PixFmt::YUV420: {
            cv::Mat yuv(m_height * 3 / 2, m_width, CV_8UC1, data, m_stride);
            cv::cvtColor(yuv, frame, cv::COLOR_YUV2BGR_I420);
            break;
        }
        case PixFmt::YUYV: {
            cv::Mat yuv(m_height, m_width, CV_8UC2, data, m_stride);
            cv::cvtColor(yuv, frame, cv::COLOR_YUV2BGR_YUYV);
            break;
        }
        default:
            request->reuse(Request::ReuseBuffers);
            m_camera->queueRequest(request);
            return;
    }

    // Store frame for read()
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_latestFrame = std::move(frame);
        m_newFrame = true;
    }
    m_frameCv.notify_one();

    // Re-queue the request for continuous capture
    request->reuse(Request::ReuseBuffers);
    m_camera->queueRequest(request);
}

// ---------- Read Frame ----------

bool LibcameraSource::read(cv::Mat& frame) {
    if (!m_active) return false;

    std::unique_lock<std::mutex> lock(m_frameMutex);
    // Wait up to 500ms for a new frame
    if (!m_frameCv.wait_for(lock, std::chrono::milliseconds(500), [this] { return m_newFrame; })) {
        return false;  // Timeout
    }

    frame = m_latestFrame.clone();
    m_newFrame = false;
    return !frame.empty();
}

// ---------- Close ----------

void LibcameraSource::close() {
    if (m_camera && m_active) {
        m_camera->stop();
    }
    m_active = false;

    m_requests.clear();

    // Unmap buffers
    for (auto& [fb, mapped] : m_mappedBuffers) {
        if (mapped.data && mapped.data != MAP_FAILED) {
            munmap(mapped.data, mapped.length);
        }
    }
    m_mappedBuffers.clear();

    if (m_allocator) {
        if (m_stream) m_allocator->free(m_stream);
        m_allocator.reset();
    }

    if (m_camera) {
        m_camera->requestCompleted.disconnect(this);
        m_camera->release();
        m_camera.reset();
    }

    if (m_cameraManager) {
        m_cameraManager->stop();
        m_cameraManager.reset();
    }

    m_stream = nullptr;
    m_pixFmt = PixFmt::Unknown;
}

// ---------- Query ----------

bool LibcameraSource::isOpened() const {
    return m_active;
}

double LibcameraSource::getFPS() const {
    return m_fps;
}

#endif // HAVE_LIBCAMERA
