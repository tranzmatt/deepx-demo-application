#ifndef SYSTEMMONITOR_H
#define SYSTEMMONITOR_H

#include <QObject>
#include <QTimer>
#include <QMutex>
#include <vector>
#include <QString>
#include <dxrt/ipc_wrapper/ipc_client_wrapper.h>

class SystemMonitor : public QObject {
    Q_OBJECT
public:
    explicit SystemMonitor(QObject* parent = nullptr);
    ~SystemMonitor();

    // Thread-safe getters
    double getCpuUsage() const;
    double getNpuUsage() const;
    double getCpuTemp() const;
    double getNpuTemp() const;

    struct NpuCoreStats {
        int id;
        double utilization;
        int temp;
        int voltage;
        int clock;
    };

    struct NpuDeviceStats {
        int id;
        QString variant;
        QString pcieBus;
        QString firmware;
        uint64_t usedMem;
        uint64_t totalMem;
        std::vector<NpuCoreStats> cores;
    };

    std::vector<NpuDeviceStats> getNpuDevices() const;

public slots:
    void start(int intervalMs = 10000);
    void stop();

signals:
    void statsUpdated();

private slots:
    void updateStats();

private:
    QTimer* m_timer;
    
    // Cached values (protected by mutex)
    mutable QMutex m_mutex;
    double m_cpuUsage;
    double m_npuUsage;
    double m_cpuTemp;
    double m_npuTemp;
    std::vector<NpuDeviceStats> m_npuDevices;

    dxrt::IPCClientWrapper m_ipcClient;
    bool m_initialized;
    bool m_ipcReady;
    bool m_staticInfoLoaded;
    std::vector<NpuDeviceStats> m_cachedDevices;
    
    // Internal counters to prevent piling up
    bool m_statsFetchInProgress;

    // Helper to read CPU usage from /proc/stat
    struct CpuData {
        unsigned long long user, nice, system, idle;
    };
    CpuData m_prevCpuData;
    CpuData readCpuData();
    
    void readCpuTemp();
    void readNpuStats();
};

#endif // SYSTEMMONITOR_H
