#ifndef JETSON_DETECT_MODEL_H
#define JETSON_DETECT_MODEL_H
#include "common.h"
#include "framework/framework.h"
#include "framework/onnx.h"
#include "config.h"

#ifdef USE_TENSORRT
    #include "framework/tensorrt.h"
#endif

class Model
{
public:
    explicit Model(const std::string &model_path);
    explicit Model(const std::string &model_path, cv::Size input_size);
    virtual ~Model() {};
    void detect(const cv::Mat &image, std::vector<det::Object> &objs);
protected:
    virtual void postprocess(const std::vector<void*> output, std::vector<det::Object> &objs) = 0;

    cv::Size m_input_size_ = {640, 640};
    det::PreParam pparam;
    std::shared_ptr<BaseFramework> framework;
};

#endif // JETSON_DETECT_MODEL_H