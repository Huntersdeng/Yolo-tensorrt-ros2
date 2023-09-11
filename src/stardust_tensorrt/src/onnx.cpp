#include "framework/onnx.h"

using namespace cv;

ONNXFramework::ONNXFramework(std::string model_path)
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    bool isGPU = true;
    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    #ifdef _WIN32
        std::wstring w_modelPath = utils::charToWstring(model_path.c_str());
        session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
    #else
        session = Ort::Session(env, model_path.c_str(), sessionOptions);
    #endif

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    this->isDynamicInputShape = false;
    // checking if width and height are dynamic
    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
    {
        std::cout << "Dynamic input shape" << std::endl;
        this->isDynamicInputShape = true;
    }

    for (auto shape : inputTensorShape)
        std::cout << "Input shape: " << shape << std::endl;

    
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());
    std::cout << "Input name: " << inputNames[0] << std::endl;

    int output_num = session.GetOutputCount();
    for (int i = 0; i < output_num; i++) {
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        outputNames.push_back(outputNodeNameAllocatedStrings.back().get());
        std::cout << "Output name: " << outputNames[i] << std::endl;
        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
        size_t output_count = outputTypeInfo.GetTensorTypeAndShapeInfo().GetElementCount();
        std::cout << "Output count: " << output_count << std::endl;

        float* temp_output_ptr = (float*)malloc(sizeof(float) * output_count);
        assert(temp_output_ptr != nullptr);
        temp_output_ptrs.push_back(temp_output_ptr);
    }
}

ONNXFramework::~ONNXFramework()
{
    for(size_t i = 0; i < temp_output_ptrs.size(); i++) {
        delete[] temp_output_ptrs[i];
    }
}

void ONNXFramework::forward(const cv::Mat &image, std::vector<void *> &output)
{
    output.clear();
    const int height = image.size[2];
    const int width = image.size[3];
    std::vector<int64_t> inputTensorShape {1, 3, height, width};
    float const* blob = image.ptr<float>();

    size_t inputTensorSize = 1 * 3 * height * width;
    std::cout << "Input tensor shape: " << inputTensorSize << std::endl;

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputTensorShape.data(), inputTensorShape.size()
    ));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              inputNames.data(),
                                                              inputTensors.data(),
                                                              inputNames.size(),
                                                              outputNames.data(),
                                                              outputNames.size());
    
    for (int i = 0; i < outputTensors.size(); ++i){
        auto* rawOutput = outputTensors[i].GetTensorData<float>();
        size_t count = outputTensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
        std::cout << "count: " << count << std::endl;
        memcpy(temp_output_ptrs[i], rawOutput, sizeof(float) * count);
    }
    for (const auto& temp_output_ptr: temp_output_ptrs) {
        output.push_back((void*) temp_output_ptr);
    }
}