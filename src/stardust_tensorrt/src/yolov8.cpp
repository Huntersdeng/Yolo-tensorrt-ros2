#include "stardust_tensorrt/yolov8.h"

using namespace det;

YOLOv8::YOLOv8(const std::string &engine_file_path) : Model(engine_file_path)
{
}

YOLOv8::~YOLOv8()
{
    std::cout << "Destruct yolov8" << std::endl;
}

void YOLOv8::preprocess(const cv::Mat &image)
{
    copy_from_Mat(image);
}

void YOLOv8::postprocess(std::vector<Object> &objs)
{
    objs.clear();
    int *num_dets = static_cast<int *>(this->host_ptrs[0]);
    auto *boxes = static_cast<float *>(this->host_ptrs[1]);
    auto *scores = static_cast<float *>(this->host_ptrs[2]);
    int *labels = static_cast<int *>(this->host_ptrs[3]);
    auto &dw = this->pparam.dw;
    auto &dh = this->pparam.dh;
    auto &width = this->pparam.width;
    auto &height = this->pparam.height;
    auto &ratio = this->pparam.ratio;
    for (int i = 0; i < num_dets[0]; i++)
    {
        float *ptr = boxes + i * 4;

        float x0 = *ptr++ - dw;
        float y0 = *ptr++ - dh;
        float x1 = *ptr++ - dw;
        float y1 = *ptr - dh;

        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = *(scores + i);
        obj.label = *(labels + i);
        objs.push_back(obj);
    }
}

void YOLOv8::detect(const cv::Mat &image, std::vector<det::Object> &objs)
{
    this->preprocess(image);
    this->infer();
    this->postprocess(objs);
}