#include "model/yolov5.h"

using namespace det;

YOLOv5::YOLOv5(const std::string &model_path) : Model(model_path)
{
}

YOLOv5::YOLOv5(const std::string &model_path, cv::Size size, float conf_thres, float nms_thres, int class_num) : Model(model_path, size),
                                                                                                        m_conf_thres_(conf_thres),
                                                                                                        m_nms_thres_(nms_thres),
                                                                                                        m_class_num_(class_num)

{
}

YOLOv5::~YOLOv5()
{
    std::cout << "Destruct yolov5" << std::endl;
}

void YOLOv5::postprocess(const std::vector<void*> output, std::vector<Object> &objs)
{
    objs.clear();
    float *outputs = static_cast<float *>(output[0]);

    int strides[3] = {8, 16, 32};
    int grid_num = 0;
    for (int i = 0; i < 3; i++)
    {
        grid_num += (m_input_size_.width / strides[i]) * (m_input_size_.height / strides[i]) * 3;
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