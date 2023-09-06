#include "model/model.h"

Model::Model(const std::string &model_path)
{
    std::string suffixStr = model_path.substr(model_path.find_last_of('.') + 1);
    if (suffixStr == "onnx")
    {
        framework = std::make_shared<ONNXFramework>(model_path);
    } else if (suffixStr == "engine")
    {
        framework = std::make_shared<TensorRTFramework>(model_path);
    }else 
    {
        framework = std::make_shared<ONNXFramework>(model_path);
    }
}