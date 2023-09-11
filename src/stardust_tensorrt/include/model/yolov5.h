#ifndef JETSON_DETECT_YOLOV5_H
#define JETSON_DETECT_YOLOV5_H
#include "model/model.h"
#include "fstream"

class YOLOv5 : public Model
{
public:
    explicit YOLOv5(const std::string &model_path);
    YOLOv5(const std::string &model_path, cv::Size size, float conf_thres, float nms_thres, int class_num);
    ~YOLOv5();

protected:
    void postprocess(const std::vector<void*> output, std::vector<det::Object> &objs) ;

private:
    float m_conf_thres_ = 0.25;
    float m_nms_thres_ = 0.65;
    int m_class_num_ = 80;
};

#endif // JETSON_DETECT_YOLOV5_H