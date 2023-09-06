#ifndef JETSON_DETECT_ONNX_H
#define JETSON_DETECT_ONNX_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "framework/framework.h"
#include <fstream>
#include <string>

class ONNXFramework: public BaseFramework
{
public:
    ONNXFramework(std::string model_path);
    ~ONNXFramework() {};
    void forward(const cv::Mat &image, std::vector<void *> &output) override;
private:
    cv::dnn::Net model;
};

#endif