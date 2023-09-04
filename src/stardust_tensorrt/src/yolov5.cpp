#include "stardust_tensorrt/yolov5.h"

using namespace det;

YOLOv5::YOLOv5(const std::string &engine_file_path) : Model(engine_file_path)
{
}

YOLOv5::YOLOv5(const std::string &engine_file_path, float conf_thres, float nms_thres, int class_num) : Model(engine_file_path),
                                                                                                        m_conf_thres_(conf_thres),
                                                                                                        m_nms_thres_(nms_thres),
                                                                                                        m_class_num_(class_num)

{
}

YOLOv5::~YOLOv5()
{
    std::cout << "Destruct yolov5" << std::endl;
}

void YOLOv5::preprocess(const cv::Mat &image)
{
    copy_from_Mat(image);
}

void YOLOv5::postprocess(std::vector<Object> &objs)
{
    objs.clear();
    float *outputs = static_cast<float *>(this->host_ptrs[0]);

    auto &in_binding = this->input_bindings[0];
    int strides[3] = {8, 16, 32};
    int grid_num = 0;
    for (int i = 0; i < 3; i++)
    {
        grid_num += (in_binding.dims.d[3] / strides[i]) * (in_binding.dims.d[2] / strides[i]) * 3;
    }

    auto &dw = this->pparam.dw;
    auto &dh = this->pparam.dh;
    auto &width = this->pparam.width;
    auto &height = this->pparam.height;
    auto &ratio = this->pparam.ratio;
    std::cout << "dw: " << dw << ", dh: " << dh << ", ratio: " << ratio << std::endl;
    float *output_ptr = outputs;
    for (int i = 0; i < grid_num; i++)
    {
        float w = output_ptr[2];
        float h = output_ptr[3];
        float x = output_ptr[0] - dw;
        float y = output_ptr[1] - dh;

        x = clamp(x * ratio, 0.f, width);
        y = clamp(y * ratio, 0.f, height);
        w = clamp(w * ratio, 0.f, width);
        h = clamp(h * ratio, 0.f, height);

        float conf = output_ptr[4];
        int class_id = -1;
        float cls_conf = 0.0;
        if (conf > m_conf_thres_)
        {
            for (int j = 0; j < m_class_num_; j++)
            {
                if (output_ptr[5 + j] > cls_conf)
                {
                    cls_conf = output_ptr[5 + j];
                    class_id = j;
                }
            }
            if (cls_conf > m_conf_thres_)
            {
                det::Object obj;
                obj.rect = cv::Rect(x - w / 2, y - h / 2, w, h);
                obj.label = class_id;
                obj.prob = cls_conf * conf;
                objs.push_back(obj);
            }
        }
        output_ptr += (m_class_num_ + 5);
    }
    nms(objs, m_nms_thres_);
}

void YOLOv5::detect(const cv::Mat &image, std::vector<det::Object> &objs)
{
    auto start = std::chrono::system_clock::now();
    this->preprocess(image);
    auto end = std::chrono::system_clock::now();
    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    std::cout << "Preprocess costs " << tc << " ms" << std::endl;

    start = std::chrono::system_clock::now();
    this->infer();
    end = std::chrono::system_clock::now();
    tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    std::cout << "Inference costs " << tc << " ms" << std::endl;

    start = std::chrono::system_clock::now();
    this->postprocess(objs);
    end = std::chrono::system_clock::now();
    tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    std::cout << "Postprocess costs " << tc << " ms" << std::endl;
}