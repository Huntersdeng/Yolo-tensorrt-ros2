#ifndef JETSON_DETECT_ONNX_H
#define JETSON_DETECT_ONNX_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "framework/framework.h"
#include <fstream>
#include <string>

class ONNXFramework: public BaseFramework
{
public:
    ONNXFramework(std::string model_path);
    ~ONNXFramework();
    void forward(const cv::Mat &image, std::vector<void *> &output) override;
private:
    void preprocessing(const cv::Mat &image, float*& blob);
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};
    bool isDynamicInputShape;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<float*> temp_output_ptrs;
};

#endif