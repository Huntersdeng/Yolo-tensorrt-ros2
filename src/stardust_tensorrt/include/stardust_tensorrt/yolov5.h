#ifndef JETSON_DETECT_YOLOV5_H
#define JETSON_DETECT_YOLOV5_H
#include "NvInferPlugin.h"
#include "stardust_tensorrt/model.h"
#include "fstream"

class YOLOv5: public Model {
public:
    explicit YOLOv5(const std::string& engine_file_path);
    YOLOv5(const std::string& engine_file_path, float conf_thres, float nms_thres, int class_num);
    ~YOLOv5();
    void detect(const cv::Mat &image, std::vector<det::Object>& objs) override;
protected:
    void preprocess(const cv::Mat &image) override;
    void postprocess(std::vector<det::Object>& objs) override;

private:
    float m_conf_thres_ = 0.25;
    float m_nms_thres_ = 0.65;
    int m_class_num_ = 80;
};

#endif  // JETSON_DETECT_YOLOV5_H