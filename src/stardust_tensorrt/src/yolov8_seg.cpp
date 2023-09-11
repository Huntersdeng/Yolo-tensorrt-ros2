#include "model/yolov8_seg.h"

using namespace det;

YOLOv8Seg::YOLOv8Seg(const std::string &engine_file_path) : Model(engine_file_path)
{
}

YOLOv8Seg::YOLOv8Seg(const std::string &engine_file_path,
                     cv::Size input_size,  float score_thres, float iou_thres, 
                     cv::Size seg_size, int seg_channels) : Model(engine_file_path, input_size), 
                                                                             m_seg_size_(seg_size), m_seg_channels_(seg_channels),
                                                                             m_score_thres_(score_thres), m_iou_thres_(iou_thres)
{
}

YOLOv8Seg::~YOLOv8Seg()
{
    std::cout << "Destruct yolov8" << std::endl;
}

void YOLOv8Seg::postprocess(const std::vector<void *> output, std::vector<Object> &objs)
{
    objs.clear();
    auto seg_h = m_seg_size_.height;
    auto seg_w = m_seg_size_.width;
    auto input_h = m_input_size_.height;
    auto input_w = m_input_size_.width;
    auto num_anchors = 8400;
    auto num_channels = 38;

    auto &dw = this->pparam.dw;
    auto &dh = this->pparam.dh;
    auto &width = this->pparam.width;
    auto &height = this->pparam.height;
    auto &ratio = this->pparam.ratio;

    auto *outputs = static_cast<float *>(output[0]);
    cv::Mat protos = cv::Mat(m_seg_channels_, seg_h * seg_w, CV_32F, static_cast<float *>(output[1]));

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> mask_confs;
    std::vector<int> indices;

    for (int i = 0; i < num_anchors; i++)
    {
        float *ptr = outputs + i * num_channels;
        float score = *(ptr + 4);
        if (score > m_score_thres_)
        {
            float x0 = *ptr++ - dw;
            float y0 = *ptr++ - dh;
            float x1 = *ptr++ - dw;
            float y1 = *ptr++ - dh;

            x0 = clamp(x0 * ratio, 0.f, width);
            y0 = clamp(y0 * ratio, 0.f, height);
            x1 = clamp(x1 * ratio, 0.f, width);
            y1 = clamp(y1 * ratio, 0.f, height);

            int label = *(++ptr);
            cv::Mat mask_conf = cv::Mat(1, m_seg_channels_, CV_32F, ++ptr);
            mask_confs.push_back(mask_conf);
            labels.push_back(label);
            scores.push_back(score);
            bboxes.push_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));
        }
    }
    cv::dnn::NMSBoxes(bboxes, scores, m_score_thres_, m_iou_thres_, indices);

    cv::Mat masks;
    int cnt = 0;
    for (auto &i : indices)
    {
        if (cnt >= topk)
        {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.prob = scores[i];
        masks.push_back(mask_confs[i]);
        objs.push_back(obj);
        cnt += 1;
    }
    if (masks.empty())
    {
        // masks is empty
    }
    else
    {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(), {seg_w, seg_h});

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        int scale_dw = dw / input_w * seg_w;
        int scale_dh = dh / input_h * seg_h;

        cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

        for (int i = 0; i < indices.size(); i++)
        {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(dest, mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].rect) > 0.5f;
        }
    }
}