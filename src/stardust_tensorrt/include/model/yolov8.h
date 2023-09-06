//
// Created by ubuntu on 3/16/23.
//
#ifndef JETSON_DETECT_YOLOV8_H
#define JETSON_DETECT_YOLOV8_H
#include "NvInferPlugin.h"
#include "model/model.h"
#include "fstream"

class YOLOv8 : public Model
{
public:
    explicit YOLOv8(const std::string &engine_file_path);
    ~YOLOv8();
    void detect(const cv::Mat &image, std::vector<det::Object> &objs) override;

protected:
    void postprocess(const std::vector<void*> output, std::vector<det::Object> &objs);

    cv::Size m_input_size_ = {640, 640};
};

#endif // JETSON_DETECT_YOLOV8_H