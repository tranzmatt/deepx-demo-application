#include "MainWindow.h"
#include <QSplitter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QThread>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QStatusBar>
#include <QMessageBox>
#include <QCoreApplication>
#include <QHeaderView>
#include <QDebug>
#include <QIcon>
#include <QMouseEvent>
#include "AdapterFactory.h"

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    initUi();
    
    // Connect VideoWidget mouse clicks
    connect(m_videoWidget, &VideoWidget::mouseClicked, this, [this](int x, int y) {
        if (m_currentAdapter) {
            m_currentAdapter->handleMouseEvent(x, y, 0); // 0 for click
        }
    });

    // Robust path finding
    QDir dir(QCoreApplication::applicationDirPath());
    QString configPath;
    
    // Check ./src/configs
    if (dir.exists("src/configs")) {
        configPath = dir.absoluteFilePath("src/configs");
    } 
    // Check ../src/configs (running from build)
    else if (dir.cdUp() && dir.exists("src/configs")) {
        configPath = dir.absoluteFilePath("src/configs");
    }
    // Fallback to relative .
    else {
        configPath = "./src/configs";
    }
    
    qDebug() << "Searching for configs in:" << configPath;

    m_configLoader = new ConfigLoader(configPath, this);
    m_configLoader->scan();
    loadConfigs();

    // Start System Monitor
    m_sysMonitor = new SystemMonitor(); 
    m_monitorThread = new QThread(this);
    
    m_sysMonitor->moveToThread(m_monitorThread);
    
    // Connect cleanups
    connect(m_monitorThread, &QThread::finished, m_sysMonitor, &QObject::deleteLater);
    
    // Connect start signal (so it starts ON the thread)
    connect(m_monitorThread, &QThread::started, m_sysMonitor, [this]() {
        m_sysMonitor->start(10000);
    });
    
    // Connect updates
    connect(m_sysMonitor, &SystemMonitor::statsUpdated, this, &MainWindow::updateSystemStats, Qt::QueuedConnection);
    
    m_monitorThread->start();
}

MainWindow::~MainWindow() {
    stopAdapter();
    
    // Safe thread shutdown
    if (m_sysMonitor) {
        QMetaObject::invokeMethod(m_sysMonitor, "stop", Qt::BlockingQueuedConnection);
    }
    
    if (m_monitorThread) {
        m_monitorThread->quit();
        m_monitorThread->wait();
    }
}

void MainWindow::initUi() {
    setWindowTitle("DeepX AI Vision Platform");
    showFullScreen(); // Start fullscreen

    // Modern Dark Theme
    QString darkStyle = R"(
        QMainWindow {
            background-color: #0d1117;
        }
        QWidget {
            background-color: #161b22;
            color: #c9d1d9;
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 13px;
        }
        QLabel {
            color: #58a6ff;
            font-weight: bold;
        }
        QListWidget {
            background-color: #21262d;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 8px;
            outline: none;
        }
        QListWidget::item {
            background-color: #21262d;
            color: #c9d1d9;
            padding: 12px 16px;
            margin: 4px 0;
            border-radius: 6px;
            border-left: 3px solid transparent;
        }
        QListWidget::item:hover {
            background-color: #30363d;
            border-left: 3px solid #58a6ff;
        }
        QListWidget::item:selected {
            background-color: #204c63;
            border-left: 3px solid #58a6ff;
            color: #ffffff;
        }
        QSplitter::handle {
            background-color: #30363d;
            width: 2px;
        }
        QStatusBar {
            background-color: #161b22;
            border-top: 1px solid #30363d;
            color: #8b949e;
            font-size: 12px;
        }
        QPushButton {
            background-color: #21262d; 
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #30363d;
            border-color: #8b949e;
        }
    )";
    setStyleSheet(darkStyle);

    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    // --- Top Info Bar ---
    m_topBar = new QWidget;
    m_topBar->setFixedHeight(80); // Taller for big FPS
    m_topBar->setStyleSheet("background-color: #204c63;");
    
    QHBoxLayout* topLayout = new QHBoxLayout(m_topBar);
    topLayout->setContentsMargins(20, 10, 20, 10);
    topLayout->setSpacing(20);

    // 1. Sidebar Toggle Button
    m_toggleSidebarBtn = new QPushButton("☰");
    m_toggleSidebarBtn->setFixedSize(40, 40);
    QFont btnFont = m_toggleSidebarBtn->font();
    btnFont.setPointSize(16);
    m_toggleSidebarBtn->setFont(btnFont);
    m_toggleSidebarBtn->setStyleSheet("background-color: #21262d; color: white; border: none; border-radius: 4px;");
    connect(m_toggleSidebarBtn, &QPushButton::clicked, this, &MainWindow::toggleSidebar);
    topLayout->addWidget(m_toggleSidebarBtn);

    // 2. FPS Label (Left, Big, White) - Fixed width to prevent logo shake
    m_fpsLabel = new QLabel("FPS: --");
    m_fpsLabel->setStyleSheet("color: #ffffff; font-size: 32px; font-weight: bold; background: transparent; padding-right: 20px;");
    m_fpsLabel->setMinimumWidth(180); // Fixed width for stability
    m_fpsLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    topLayout->addWidget(m_fpsLabel);

    topLayout->addStretch(); // Spacer left

    // 3. Logo (Center)
    QLabel* logoLabel = new QLabel;
    QPixmap logo("../src/images/sixfab.png");
    if (!logo.isNull()) {
        logoLabel->setPixmap(logo.scaledToHeight(60, Qt::SmoothTransformation));
    } else {
        logoLabel->setText("SIXFAB");
        logoLabel->setStyleSheet("color: #ffffff; font-size: 24px; font-weight: bold;");
    }
    logoLabel->setStyleSheet("background: transparent;");
    topLayout->addWidget(logoLabel);

    topLayout->addStretch(); // Spacer right

    // 4. Stats Grid (Right, 2x2)
    QWidget* statsWidget = new QWidget;
    statsWidget->setStyleSheet("background: transparent;");
    QGridLayout* statsGrid = new QGridLayout(statsWidget);
    statsGrid->setContentsMargins(0, 0, 0, 0);
    statsGrid->setSpacing(10);

    // Helper to create stat label
    auto createStatLabel = [](QString text) {
        QLabel* l = new QLabel(text);
        l->setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;");
        return l;
    };

    m_cpuUsageLabel = createStatLabel("CPU Usage: --%");
    m_npuUsageLabel = createStatLabel("NPU Usage: --%");

    statsGrid->addWidget(m_npuUsageLabel, 0, 0);
    statsGrid->addWidget(m_cpuUsageLabel, 1, 0);

    topLayout->addWidget(statsWidget);


    mainLayout->addWidget(m_topBar);

    // --- Main Content (Sidebar + Video) ---
    QSplitter* splitter = new QSplitter(Qt::Horizontal, centralWidget);
    
    m_leftPanel = new QWidget;
    m_leftPanel->setMinimumWidth(280);
    m_leftPanel->setMaximumWidth(400);
    QVBoxLayout* leftLayout = new QVBoxLayout(m_leftPanel);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->setSpacing(0);
    
    splitter->addWidget(m_leftPanel);
    
    // AI Configurations Layout
    QWidget* hubContent = new QWidget;
    QVBoxLayout* hubLayout = new QVBoxLayout(hubContent);
    hubLayout->setContentsMargins(16, 16, 16, 16);
    hubLayout->setSpacing(12);

    QLabel* titleLabel = new QLabel("🎯 AI Configurations");
    titleLabel->setStyleSheet("font-size: 16px; color: #ffffff; padding: 8px 0; font-weight: bold;");
    hubLayout->addWidget(titleLabel);
    
    m_configList = new QListWidget;
    m_configList->setStyleSheet(R"(
        QListWidget {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            color: #c9d1d9;
            padding: 5px;
        }
        QListWidget::item {
            padding: 6px 10px;
            border-bottom: 1px solid #30363d;
            min-height: 70px;
        }
        QListWidget::item:selected {
            background-color: #204c63;
            color: #ffffff;
            border-radius: 4px;
        }
    )");
    connect(m_configList, &QListWidget::itemClicked, this, &MainWindow::onConfigSelected);
    hubLayout->addWidget(m_configList, 1); // 100% height
    
    // Source Selection (Hidden by default)
    m_sourceGroupWidget = new QWidget;
    m_sourceGroupWidget->setStyleSheet(R"(
        QWidget {
            background-color: #21262d;
            border: 2px solid #30363d;
            border-radius: 12px;
        }
        QRadioButton {
            color: #c9d1d9;
            font-size: 13px;
            font-weight: bold;
            padding: 5px;
        }
    )");
    QGridLayout* sourceLayout = new QGridLayout(m_sourceGroupWidget);
    sourceLayout->setContentsMargins(10, 10, 10, 10);
    sourceLayout->setSpacing(5);
    hubLayout->addWidget(m_sourceGroupWidget);
    m_sourceGroupWidget->hide();

    leftLayout->addWidget(hubContent, 1);
    
    m_sourceButtonGroup = new QButtonGroup(this);
    // Connect to the new signal signature
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    connect(m_sourceButtonGroup, &QButtonGroup::idClicked, this, &MainWindow::onSourceIndexChanged);
#else
    connect(m_sourceButtonGroup, static_cast<void(QButtonGroup::*)(int)>(&QButtonGroup::buttonClicked), this, [this](int id){
        this->onSourceIndexChanged(id);
    });
#endif
    
    // Right Panel (Video with Side Banners)
    QWidget* videoContainer = new QWidget;
    videoContainer->setStyleSheet("background-color: #204c63;");
    QHBoxLayout* videoLayout = new QHBoxLayout(videoContainer);
    videoLayout->setContentsMargins(0, 0, 0, 0);
    videoLayout->setSpacing(0);

    QWidget* leftBanner = new QWidget;
    leftBanner->setStyleSheet("background-color: #204c63;");
    leftBanner->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    videoLayout->addWidget(leftBanner);

    m_videoWidget = new VideoWidget;
    m_videoWidget->setStyleSheet("background-color: #0d1117;");
    m_videoWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    videoLayout->addWidget(m_videoWidget, 1);

    QWidget* rightBanner = new QWidget;
    rightBanner->setStyleSheet("background-color: #204c63;");
    rightBanner->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    videoLayout->addWidget(rightBanner);

    splitter->addWidget(videoContainer);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 5);
    mainLayout->addWidget(splitter);
}

void MainWindow::loadConfigs() {
    m_configList->clear();
    auto configs = m_configLoader->getConfigs();
    for (const auto& cfg : configs) {
        QListWidgetItem* item = new QListWidgetItem();
        item->setData(Qt::UserRole, cfg.filePath);

        // Custom two-line widget
        QWidget* itemWidget = new QWidget;
        itemWidget->setAttribute(Qt::WA_TranslucentBackground);
        QVBoxLayout* itemLayout = new QVBoxLayout(itemWidget);
        itemLayout->setContentsMargins(8, 6, 8, 6);
        itemLayout->setSpacing(4);

        QLabel* descLabel = new QLabel(cfg.description);
        descLabel->setStyleSheet("color: #c9d1d9; font-size: 13px; font-weight: bold; background: transparent;");
        descLabel->setWordWrap(true);
        descLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
        itemLayout->addWidget(descLabel);

        if (!cfg.subDescription.isEmpty()) {
            QLabel* subLabel = new QLabel(cfg.subDescription);
            subLabel->setStyleSheet("color: #8b949e; font-size: 10px; font-weight: normal; background: transparent;");
            subLabel->setWordWrap(true);
            subLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
            itemLayout->addWidget(subLabel);
        }

        itemLayout->addStretch();
        itemWidget->setMinimumHeight(70);
        item->setSizeHint(QSize(280, 70));
        m_configList->addItem(item);
        m_configList->setItemWidget(item, itemWidget);
    }
}

void MainWindow::onConfigSelected(QListWidgetItem* item) {
    QString path = item->data(Qt::UserRole).toString();
    auto configs = m_configLoader->getConfigs();
    
    for (const auto& cfg : configs) {
        if (cfg.filePath == path) {
            startAdapter(cfg);
            break;
        }
    }
}

void MainWindow::stopAdapter() {
    if (m_currentAdapter) {
        m_currentAdapter->stop();
        disconnect(m_currentAdapter.get(), nullptr, this, nullptr);
        m_currentAdapter.reset();
        m_fpsLabel->setText("FPS: --"); // Reset FPS on stop
    }
}

void MainWindow::startAdapter(const AdapterConfig& config) {
    stopAdapter();
    
    m_currentAdapter = AdapterFactory::createAdapter(config.adapterType, config.sourceCount);
    if (!m_currentAdapter) {
        QMessageBox::warning(this, "Error", "Unknown adapter type: " + config.adapterType);
        return;
    }

    connect(m_currentAdapter.get(), &IAdapter::frameReady, this, &MainWindow::onAdapterFrame);
    connect(m_currentAdapter.get(), &IAdapter::errorOccurred, this, &MainWindow::onAdapterError);
    connect(m_currentAdapter.get(), &IAdapter::fpsUpdate, this, &MainWindow::onFpsUpdate);

    if (!m_currentAdapter->loadConfig(config.json)) {
        m_currentAdapter.reset();
        return;
    }

    m_currentAdapter->start();
    m_currentConfig = config;

    if (config.adapterType != "multichannel") {
        QJsonArray sources = config.json["input"].toObject()["sources"].toArray();
        rebuildSourceRadioButtons(sources);
    } else {
        m_sourceGroupWidget->hide();
    }
}

// ... (other methods)

void MainWindow::toggleSidebar() {
    if (m_leftPanel) {
        m_leftPanel->setVisible(!m_leftPanel->isVisible());
    }
}

void MainWindow::onAdapterFrame(const QImage& frame, const DetectionResult& result) {
    m_videoWidget->updateFrame(frame, result);
}

void MainWindow::onAdapterError(const QString& msg) {
    // m_statusLabel->setText("Error: " + msg);
    qWarning() << "Adapter Error:" << msg;
}

void MainWindow::onFpsUpdate(double fps) {
    m_currentFps = fps;
    // Update Big FPS Label
    if (fps <= 0) {
        m_fpsLabel->setText("FPS: --");
    } else {
        m_fpsLabel->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
    }
}

void MainWindow::updateSystemStats() {
    if (!m_sysMonitor) return;

    double cpuUsage = m_sysMonitor->getCpuUsage();
    double npuUsage = m_sysMonitor->getNpuUsage();

    m_cpuUsageLabel->setText(QString("CPU Usage: %1%").arg(cpuUsage, 0, 'f', 1));
    m_npuUsageLabel->setText(QString("NPU Usage: %1%").arg(npuUsage, 0, 'f', 1));

    // Update NPU Hover Tooltip
    auto devices = m_sysMonitor->getNpuDevices();
    if (!devices.empty()) {
        const auto& dev = devices[0];
        
        QString tooltip = "<html><body style='background-color: #1c2128; color: #c9d1d9; font-family: sans-serif; padding: 10px;'>";
        tooltip += QString("<h3 style='color: #58a6ff; margin: 0 0 10px 0;'>🚀 NPU: %1</h3>").arg(dev.variant);
        
        // Memory Section
        if (dev.totalMem > 0) {
            auto formatBytes = [](uint64_t bytes) {
                if (bytes >= 1024*1024*1024) return QString("%1 GB").arg((double)bytes / (1024*1024*1024), 0, 'f', 2);
                if (bytes >= 1024*1024) return QString("%1 MB").arg((double)bytes / (1024*1024), 0, 'f', 1);
                return QString("%1 KB").arg((double)bytes / 1024, 0, 'f', 0);
            };
            int memPercent = (int)(dev.usedMem * 100 / dev.totalMem);
            tooltip += QString("<p style='margin: 5px 0;'><b>Memory:</b> %1 / %2 (%3%)</p>")
                        .arg(formatBytes(dev.usedMem))
                        .arg(formatBytes(dev.totalMem))
                        .arg(memPercent);
        }

        // Core Section
        tooltip += "<table style='width: 100%; border-collapse: collapse; margin-top: 10px;'>";
        tooltip += "<tr style='background-color: #30363d;'><th style='padding: 4px;'>Core</th><th style='padding: 4px;'>Usage</th><th style='padding: 4px;'>Temp</th></tr>";
        
        for (const auto& core : dev.cores) {
            QString color = "#3fb950"; // Green
            if (core.utilization > 80) color = "#f85149"; // Red
            else if (core.utilization > 40) color = "#d29922"; // Yellow

            tooltip += QString("<tr>")
                        + QString("<td style='padding: 4px; border-bottom: 1px solid #30363d;'>Core %1</td>").arg(core.id)
                        + QString("<td style='padding: 4px; border-bottom: 1px solid #30363d; color: %1; font-weight: bold;'>%2%</td>").arg(color).arg(core.utilization, 0, 'f', 1)
                        + QString("<td style='padding: 4px; border-bottom: 1px solid #30363d;'>%1°C</td>").arg(core.temp)
                        + "</tr>";
        }
        tooltip += "</table>";
        tooltip += "</body></html>";
        
        m_npuUsageLabel->setToolTip(tooltip);
        m_npuUsageLabel->setCursor(Qt::PointingHandCursor);
    }
}

void MainWindow::rebuildSourceRadioButtons(const QJsonArray& sources) {
    if (sources.isEmpty()) {
        m_sourceGroupWidget->hide();
        return;
    }

    for (auto rb : m_sourceRadioButtons) {
        m_sourceButtonGroup->removeButton(rb);
        delete rb;
    }
    m_sourceRadioButtons.clear();

    QGridLayout* layout = qobject_cast<QGridLayout*>(m_sourceGroupWidget->layout());
    QMap<QString, int> typeCounts;
    
    int row = 0;
    int col = 0;
    
    for (int i = 0; i < sources.size(); ++i) {
        QJsonObject srcObj = sources[i].toObject();
        QString type = srcObj["type"].toString();
        
        QString baseName;
        if (type == "video") baseName = "Video";
        else if (type == "libcamera") baseName = "Raspberry Pi Cam";
        else if (type == "camera") baseName = "Other Camera";
        else if (type == "rtsp") baseName = "RTSP Stream";
        else baseName = type;
        
        typeCounts[baseName]++;
        int count = typeCounts[baseName];
        QString label = baseName;
        if (count > 1) label += QString(" %1").arg(count);
        
        QRadioButton* rb = new QRadioButton(label);
        m_sourceRadioButtons.append(rb);
        m_sourceButtonGroup->addButton(rb, i);
        layout->addWidget(rb, row, col);
        
        col++;
        if (col > 1) {
            col = 0;
            row++;
        }
    }
    
    // Legacy fallback check
    if (sources.size() == 1) {
        QJsonObject firstSource = sources[0].toObject();
        if (firstSource.contains("secondary_path")) {
             QString secType = firstSource["secondary_type"].toString(firstSource["type"].toString());
             QString baseName;
             if (secType == "video") baseName = "Video";
             else if (secType == "libcamera") baseName = "Raspberry Pi Cam";
             else if (secType == "camera") baseName = "Other Camera";
             else if (secType == "rtsp") baseName = "RTSP Stream";
             else baseName = secType;
             
             typeCounts[baseName]++;
             QString label = baseName;
             if (typeCounts[baseName] > 1) label += QString(" %1").arg(typeCounts[baseName]);
             
             QRadioButton* rb = new QRadioButton(label);
             m_sourceRadioButtons.append(rb);
             m_sourceButtonGroup->addButton(rb, 1);
             layout->addWidget(rb, row, col);
        }
    }

    if (!m_sourceRadioButtons.isEmpty()) {
        m_sourceRadioButtons[0]->setChecked(true);
        m_currentAdapter->setSourceIndex(0);
        m_sourceGroupWidget->show();
    } else {
        m_sourceGroupWidget->hide();
    }
}

void MainWindow::onSourceIndexChanged(int index) {
    if (m_currentAdapter) {
        m_currentAdapter->setSourceIndex(index);
    }
}


