#ifndef JETSON_DETECT_MODEL_H
#define JETSON_DETECT_MODEL_H
#include "common.h"
#include "framework/framework.h"
#include "framework/onnx.h"

#ifdef USE_TENSORRT
    #include "framework/tensorrt.h"
#endif

class Model
{
public:
    explicit Model(const std::string &model_path);
    virtual ~Model() {};
    virtual void detect(const cv::Mat &image, std::vector<det::Object> &objs) = 0;

    det::PreParam pparam;
    std::shared_ptr<BaseFramework> framework;
};
#endif // JETSON_DETECT_MODEL_H