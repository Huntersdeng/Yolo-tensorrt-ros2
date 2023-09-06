#include "framework/onnx.h"

using namespace cv;

ONNXFramework::ONNXFramework(std::string model_path)
{
    this->model = dnn::readNetFromONNX(model_path);
    this->model.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    this->model.setPreferableTarget(dnn::DNN_TARGET_CPU);
}

void ONNXFramework::forward(const cv::Mat &image, std::vector<void *> &output)
{
    output.clear();
    std::vector<std::string> outLayerNames = this->model.getUnconnectedOutLayersNames();
    std::vector<Mat> outs;
    this->model.setInput(image);
    this->model.forward(outs, outLayerNames);
    for (const auto &out: outs) {
        output.push_back((void*)(out.ptr<float>()));
    }
}