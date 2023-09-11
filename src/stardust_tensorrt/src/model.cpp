#include "model/model.h"


Model::Model(const std::string &model_path)
{
    std::string suffixStr = model_path.substr(model_path.find_last_of('.') + 1);
    if (suffixStr == "onnx")
    {
        framework = std::make_shared<ONNXFramework>(model_path);
    }
    #ifdef USE_TENSORRT
    else if (suffixStr == "engine")
    {
        framework = std::make_shared<TensorRTFramework>(model_path);
    }
    #endif
    else 
    {
        #ifdef USE_TENSORRT
            std::cerr << "Only support *.onnx and *.engine files" << std::endl;
        #else 
            std::cerr << "Only support *.onnx files" << std::endl;
        #endif
    }
}

Model::Model(const std::string &model_path, cv::Size input_size) : Model(model_path)
{
    m_input_size_ = input_size;
}

void Model::detect(const cv::Mat &image, std::vector<det::Object> &objs)
{
    cv::Mat nchw;
    this->pparam = letterbox(image, nchw, m_input_size_);
    std::vector<void*> output;
    this->framework->forward(nchw, output);
    postprocess(output, objs);
}