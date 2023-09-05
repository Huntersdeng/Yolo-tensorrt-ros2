#ifndef JETSON_DETECT_MODEL_H
#define JETSON_DETECT_MODEL_H
#include "NvInferPlugin.h"
#include "stardust_tensorrt/common.h"
#include "fstream"

class Model
{
public:
    explicit Model(const std::string &engine_file_path);
    virtual ~Model();
    virtual void detect(const cv::Mat &image, std::vector<det::Object> &objs) = 0;

protected:
    void make_pipe(bool warmup = true);
    virtual void preprocess(const cv::Mat &image) = 0;
    void infer();
    virtual void postprocess(std::vector<det::Object> &objs) = 0;
    void copy_from_Mat(const cv::Mat &image);
    void copy_from_Mat(const cv::Mat &image, cv::Size &size);
    void letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size);
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

#endif // JETSON_DETECT_Model_H