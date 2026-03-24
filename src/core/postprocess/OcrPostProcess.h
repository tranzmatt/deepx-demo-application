#ifndef OCRPOSTPROCESS_H
#define OCRPOSTPROCESS_H

#include "Common.h"
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <memory>

// Structure to hold raw detection results before they are converted to DetectionBox
struct OcrBox {
    std::vector<cv::Point> box; // 4 points
    float score;
};

class OcrPostProcess {
public:
    OcrPostProcess();
    ~OcrPostProcess();

    // --- Configuration ---
    void setDetThresh(float thresh) { m_detThresh = thresh; }
    void setDetBoxThresh(float thresh) { m_detBoxThresh = thresh; }
    void setUnclipRatio(float ratio) { m_unclipRatio = ratio; }
    void setRecScoreThresh(float thresh) { m_recScoreThresh = thresh; }
    
    // Load character dictionary for recognition
    bool loadCharDict(const std::string& dictPath);

    // --- Detection Post-Processing (DB Algorithm) ---
    // Takes the raw output map (1, H, W) from DB model and returns bounding boxes
    // mapData: pointer to float data of the probability map
    // mapW, mapH: dimensions of the probability map
    // inputW, inputH: dimensions of the input image (for scaling back)
    std::vector<OcrBox> postProcessDet(const float* mapData, int mapW, int mapH, int inputW, int inputH);

    // --- Recognition Post-Processing (CTC Decode) ---
    // Takes raw output from Rec model and decodes to string
    // outputData: pointer to float data (Batch, Time, NumClasses)
    // seqLen: Time dimension length
    // numClasses: Number of classes
    // Returns {text, confidence}
    std::pair<std::string, float> postProcessRec(const float* outputData, int seqLen, int numClasses);

    // --- Classification Post-Processing ---
    // Returns index (0: 0 degrees, 1: 180 degrees) and score
    std::pair<int, float> postProcessCls(const float* outputData, int numClasses);

    // Get number of classes in loaded dictionary
    int getNumClasses() const { return m_characterDict.size(); }

private:
    float m_detThresh = 0.3f;
    float m_detBoxThresh = 0.6f;
    float m_unclipRatio = 1.6f; // Slightly increased from 1.5 default for robustness
    float m_recScoreThresh = 0.5f;

    std::vector<std::string> m_characterDict;
    
    // Helper to unclip (expand) the polygon
    std::vector<cv::Point> unclip(const std::vector<cv::Point>& box, float unclipRatio);
    
    // Ordered clockwise: TL, TR, BR, BL
    void orderPoints(std::vector<cv::Point>& pts);
    
    // Helper to calculate box score from map
    float getBoxScore(const std::vector<cv::Point>& box, const cv::Mat& pred);
};

#endif // OCRPOSTPROCESS_H
