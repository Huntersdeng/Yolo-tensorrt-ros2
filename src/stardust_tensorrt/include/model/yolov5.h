#ifndef JETSON_DETECT_YOLOV5_H
#define JETSON_DETECT_YOLOV5_H
#include "NvInferPlugin.h"
#include "model/model.h"
#include "fstream"

class YOLOv5 : public Model
{
public:
    explicit YOLOv5(const std::string &model_path);
    YOLOv5(const std::string &model_path, float conf_thres, float nms_thres, int class_num, cv::Size size);
    ~YOLOv5();
    void detect(const cv::Mat &image, std::vector<det::Object> &objs) override;

protected:
    void postprocess(const std::vector<void*> output, std::vector<det::Object> &objs);

private:
    float m_conf_thres_ = 0.25;
    float m_nms_thres_ = 0.65;
    int m_class_num_ = 80;
    cv::Size m_input_size_ = {640, 640};
    det::PreParam pparam;
};

#endif // JETSON_DETECT_YOLOV5_H