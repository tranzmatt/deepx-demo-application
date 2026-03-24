#include "SystemMonitor.h"
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <QThread>
#include <QElapsedTimer>
#include <cmath>
#include <dxrt/device_pool.h>
#include <dxrt/device_task_layer.h>

SystemMonitor::SystemMonitor(QObject* parent) 
    : QObject(parent)
    , m_timer(nullptr)
    , m_cpuUsage(0.0)
    , m_npuUsage(0.0)
    , m_cpuTemp(0.0)
    , m_npuTemp(0.0)
    , m_ipcClient(dxrt::IPCDefaultType(), getpid() + 1234)
    , m_initialized(false)
    , m_ipcReady(false)
    , m_staticInfoLoaded(false)
    , m_statsFetchInProgress(false)
{
    m_prevCpuData = readCpuData();
}

SystemMonitor::~SystemMonitor() {
    stop();
}

void SystemMonitor::start(int intervalMs) {
    if (!m_initialized) {
        dxrt::DevicePool::GetInstance().InitCores();
        try {
            int ret = m_ipcClient.Initialize(false);
            if (ret == 0) {
                m_ipcReady = true;
            } else {
                qWarning() << "SystemMonitor: IPC Init failed:" << ret;
            }
        } catch (...) {
            qWarning() << "SystemMonitor: IPC Init exception";
        }
        
        m_timer = new QTimer(this);
        connect(m_timer, &QTimer::timeout, this, &SystemMonitor::updateStats);
        m_initialized = true;
    }
    m_timer->start(intervalMs);
    updateStats();
}

void SystemMonitor::stop() {
    if (m_timer) {
        m_timer->stop();
    }
}

double SystemMonitor::getCpuUsage() const { 
    QMutexLocker lock(&m_mutex);
    return m_cpuUsage; 
}

double SystemMonitor::getNpuUsage() const { 
    QMutexLocker lock(&m_mutex);
    return m_npuUsage; 
}

double SystemMonitor::getCpuTemp() const { 
    QMutexLocker lock(&m_mutex);
    return m_cpuTemp; 
}

double SystemMonitor::getNpuTemp() const { 
    QMutexLocker lock(&m_mutex);
    return m_npuTemp; 
}

std::vector<SystemMonitor::NpuDeviceStats> SystemMonitor::getNpuDevices() const {
    QMutexLocker lock(&m_mutex);
    return m_npuDevices;
}

void SystemMonitor::updateStats() {
    if (m_statsFetchInProgress) return; // Prevent piling up if it hangs
    m_statsFetchInProgress = true;

    QElapsedTimer timer;
    timer.start();

    // 1. CPU Usage
    CpuData curData = readCpuData();
    unsigned long long prevTotal = m_prevCpuData.user + m_prevCpuData.nice + m_prevCpuData.system + m_prevCpuData.idle;
    unsigned long long curTotal = curData.user + curData.nice + curData.system + curData.idle;
    
    unsigned long long totalDiff = curTotal - prevTotal;
    unsigned long long idleDiff = curData.idle - m_prevCpuData.idle;
    
    double cpuUsage = m_cpuUsage;
    if (totalDiff > 0) {
        cpuUsage = 100.0 * (1.0 - (double)idleDiff / totalDiff);
    }
    m_prevCpuData = curData;

    readCpuTemp();
    readNpuStats();

    {
        QMutexLocker lock(&m_mutex);
        m_cpuUsage = cpuUsage;
    }

    emit statsUpdated();
    m_statsFetchInProgress = false;
    // qDebug() << "SystemMonitor: Update stats took" << timer.elapsed() << "ms";
}

SystemMonitor::CpuData SystemMonitor::readCpuData() {
    CpuData data = {0, 0, 0, 0};
    QFile file("/proc/stat");
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QString line = file.readLine();
        QTextStream stream(&line);
        QString cpuLabel;
        stream >> cpuLabel >> data.user >> data.nice >> data.system >> data.idle;
    }
    return data;
}

void SystemMonitor::readCpuTemp() {
    // Try reliable thermal zone. Usually zone0 is CPU or SoC
    QFile file("/sys/class/thermal/thermal_zone0/temp");
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QString line = file.readAll().trimmed();
        double tempMilli = line.toDouble();
        QMutexLocker lock(&m_mutex);
        m_cpuTemp = tempMilli / 1000.0;
    } else {
        QMutexLocker lock(&m_mutex);
        m_cpuTemp = 0.0; // Fallback
    }
}

void SystemMonitor::readNpuStats() {
    using namespace dxrt;
    
    auto& pool = DevicePool::GetInstance();
    int deviceCount = pool.GetDeviceCount();

    if (!m_staticInfoLoaded || m_cachedDevices.size() != (size_t)deviceCount) {
        m_cachedDevices.clear();
        for (int i = 0; i < deviceCount; ++i) {
            auto devCore = pool.GetDeviceCores(i);
            if (!devCore) continue;

            NpuDeviceStats dStats;
            dStats.id = i;
            auto info = devCore->info();
            
            switch (info.variant) {
                case 100: dStats.variant = "L1"; break;
                case 101: dStats.variant = "L2"; break;
                case 102: dStats.variant = "L3"; break;
                case 103: dStats.variant = "L4"; break;
                case 200: case 201: case 202: dStats.variant = "M1"; break;
                default: dStats.variant = QString("Unknown (%2)").arg(info.variant);
            }

            dStats.firmware = QString("v%1.%2.%3")
                                .arg(info.fw_ver / 100)
                                .arg((info.fw_ver / 1) % 10)
                                .arg(info.fw_ver % 10);
            
            auto devInfo = devCore->devInfo();
            dStats.pcieBus = QString("%1:%2:%3")
                                .arg(devInfo.pcie.bus, 2, 16, QChar('0'))
                                .arg(devInfo.pcie.dev, 2, 16, QChar('0'))
                                .arg(devInfo.pcie.func, 2, 16, QChar('0'));

            dStats.totalMem = info.mem_size;
            m_cachedDevices.push_back(dStats);
        }
        m_staticInfoLoaded = true;
    }

    std::vector<NpuDeviceStats> currentDevices = m_cachedDevices;
    double totalUsage = 0.0;
    double maxTemp = 0.0;
    int coreCount = 0;

    for (auto& dStats : currentDevices) {
        auto devCore = pool.GetDeviceCores(dStats.id);
        if (!devCore) continue;

        if (m_ipcReady) {
            // DRAM Usage via IPC
            try {
                IPCClientMessage msg;
                IPCServerMessage resp;
                msg.code = REQUEST_CODE::VIEW_USED_MEMORY;
                msg.deviceId = dStats.id;
                msg.pid = getpid() + 1234; // Matches identifier
                m_ipcClient.SendToServer(resp, msg);
                dStats.usedMem = resp.data;
            } catch (...) {
                dStats.usedMem = 0;
            }
        }

        // Core Stats
        auto status = devCore->Status();
        for (int c = 0; c < 3; ++c) { // DeepX NPU variant has 3 cores
            NpuCoreStats cStats;
            cStats.id = c;
            cStats.temp = (status.temperature[c] > 200) ? 0 : status.temperature[c];
            cStats.voltage = status.voltage[c];
            cStats.clock = status.clock[c];

            if (m_ipcReady) {
                try {
                    IPCClientMessage msg;
                    IPCServerMessage resp;
                    msg.code = REQUEST_CODE::GET_USAGE;
                    msg.deviceId = dStats.id;
                    msg.data = c; 
                    msg.pid = getpid() + 1234; // Matches identifier
                    m_ipcClient.SendToServer(resp, msg);
                    cStats.utilization = (double)resp.data / 10.0; 
                } catch (...) {
                    cStats.utilization = 0.0;
                }
            } else {
                cStats.utilization = 0.0;
            }

            dStats.cores.push_back(cStats);
            totalUsage += cStats.utilization;
            if (cStats.temp > maxTemp) maxTemp = cStats.temp;
            coreCount++;
        }
    }

    {
        QMutexLocker lock(&m_mutex);
        m_npuDevices = currentDevices;
        if (coreCount > 0) {
            m_npuUsage = totalUsage / coreCount;
            m_npuTemp = maxTemp;
        } else {
            m_npuUsage = 0.0;
            m_npuTemp = 0.0;
        }
    }
}
