#ifndef JETSON_DETECT_TENSORRT_H
#define JETSON_DETECT_TENSORRT_H

#include "NvInferPlugin.h"
#include "framework/framework.h"
#include <fstream>

namespace det 
{
    struct Binding
    {
        size_t size = 1;
        size_t dsize = 1;
        nvinfer1::Dims dims;
        std::string name;
    };
}

class Logger : public nvinfer1::ILogger
{
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) : reportableSeverity(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override;
};

int get_size_by_dims(const nvinfer1::Dims &dims);

int type_to_size(const nvinfer1::DataType &dataType);

class TensorRTFramework: public BaseFramework
{
public:
    explicit TensorRTFramework(const std::string &engine_file_path);
    virtual ~TensorRTFramework();
    void forward(const cv::Mat &image, std::vector<void *> &output) override;

private:
    void make_pipe(bool warmup = true);
    void set_input(const cv::Mat &image);
    void infer();

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<det::Binding> input_bindings;
    std::vector<det::Binding> output_bindings;
    std::vector<void *> host_ptrs;
    std::vector<void *> device_ptrs;

    det::PreParam pparam;
};

#endif