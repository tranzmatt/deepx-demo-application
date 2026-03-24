#ifndef BASEADAPTER_H
#define BASEADAPTER_H

#include "IAdapter.h"
#include <thread>
#include <atomic>
#include <mutex>

class BaseAdapter : public IAdapter {
    Q_OBJECT
public:
    explicit BaseAdapter(QObject* parent = nullptr);
    virtual ~BaseAdapter();

    void start() override;
    void stop() override;
    void setSourceIndex(int index) override;

    // Helper to resolve paths
    static std::string resolvePath(const QString& p);

protected:
    virtual void processLoop() = 0;
    
    // Helper to open sources flexibly
    bool openSource(const SourceInfo& info, class VideoSource* source);
    SourceInfo getCurrentSource();

    std::atomic<bool> m_running;
    std::thread m_workerThread;
    std::mutex m_mutex;

    // Source switching state
    std::vector<SourceInfo> m_sources;
    std::atomic<int> m_sourceIndex{0};
    std::atomic<bool> m_reopenRequested{false};
    std::mutex m_sourceMutex;
};

#endif // BASEADAPTER_H
