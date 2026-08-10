#include "SystemMonitor.h"
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <QThread>
#include <QElapsedTimer>
#include <cmath>
#include <algorithm>
#include <dxrt/dxrt_cxx_api.h>

SystemMonitor::SystemMonitor(QObject* parent)
    : QObject(parent)
    , m_timer(nullptr)
    , m_cpuUsage(0.0)
    , m_npuUsage(0.0)
    , m_cpuTemp(0.0)
    , m_npuTemp(0.0)
    , m_initialized(false)
    , m_monitoringActive(false)
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
        try {
            auto& pool = dxrt::DevicePool::GetInstance();
            pool.InitCores();
            // Utilization and memory usage come from the dxrtd shared memory
            // segment; if dxrtd is not publishing, those fields stay at zero.
            m_monitoringActive = pool.GetDeviceCount() > 0 &&
                                 dxrt::DeviceStatus::GetCurrentStatus(0).IsValid();
            if (!m_monitoringActive) {
                qWarning() << "SystemMonitor: dxrtd monitoring data unavailable";
            }
        } catch (const dxrt::Exception& e) {
            qWarning() << "SystemMonitor: device pool init failed:" << e.what();
        } catch (...) {
            qWarning() << "SystemMonitor: device pool init exception";
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
    int deviceCount = 0;
    try {
        deviceCount = static_cast<int>(pool.GetDeviceCount());
    } catch (const Exception& e) {
        qWarning() << "SystemMonitor: GetDeviceCount failed:" << e.what();
        return;
    }

    if (!m_staticInfoLoaded || m_cachedDevices.size() != (size_t)deviceCount) {
        m_cachedDevices.clear();
        for (int i = 0; i < deviceCount; ++i) {
            auto status = DeviceStatus::GetCurrentStatus(i);

            NpuDeviceStats dStats;
            dStats.id = i;
            // The runtime now reports these directly as strings
            // (e.g. "M1", "2.7.4", "Gen3 X1 [01:00:00]").
            dStats.variant = QString::fromStdString(status.DeviceVariantStr());
            if (dStats.variant.isEmpty()) dStats.variant = "Unknown";

            const std::string fw = status.FirmwareVersionStr();
            dStats.firmware = fw.empty() ? QString("Unknown")
                                         : QString("v%1").arg(QString::fromStdString(fw));

            dStats.pcieBus = QString::fromStdString(status.PcieInfoStr());

            const int64_t memSize = status.MemorySize();
            dStats.totalMem = memSize > 0 ? static_cast<uint64_t>(memSize) : 0;
            dStats.usedMem = 0;

            m_cachedDevices.push_back(dStats);
        }
        m_staticInfoLoaded = true;
    }

    std::vector<NpuDeviceStats> currentDevices = m_cachedDevices;
    double totalUsage = 0.0;
    double maxTemp = 0.0;
    int coreCount = 0;

    for (auto& dStats : currentDevices) {
        auto status = DeviceStatus::GetCurrentStatus(dStats.id);

        if (m_monitoringActive) {
            // DRAM usage now comes from the dxrtd shared memory segment
            // instead of a VIEW_USED_MEMORY IPC round-trip.
            dStats.usedMem = status.GetMemoryUsed();
        }

        // Core Stats
        for (int c = 0; c < 3; ++c) { // DeepX NPU variant has 3 cores
            NpuCoreStats cStats;
            cStats.id = c;
            const int temp = status.Temperature(c);
            cStats.temp = (temp > 200) ? 0 : temp;
            cStats.voltage = static_cast<int>(status.Voltage(c));
            cStats.clock = static_cast<int>(status.NpuClock(c));

            if (m_monitoringActive) {
                // Already a 0.0-100.0 percentage; the old IPC path returned
                // tenths of a percent and had to be divided by 10.
                cStats.utilization = std::max(0.0, status.GetCoreUtilization(c));
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
