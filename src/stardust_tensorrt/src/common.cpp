#include "stardust_tensorrt/common.h"

using namespace det;

void Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept
{
    if (severity > reportableSeverity)
    {
        return;
    }
    switch (severity)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
    case nvinfer1::ILogger::Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
    case nvinfer1::ILogger::Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
    case nvinfer1::ILogger::Severity::kINFO:
        std::cerr << "INFO: ";
        break;
    default:
        std::cerr << "VERBOSE: ";
        break;
    }
    std::cerr << msg << std::endl;
}

int get_size_by_dims(const nvinfer1::Dims &dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++)
    {
        size *= dims.d[i];
    }
    return size;
}

int type_to_size(const nvinfer1::DataType &dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        return 4;
    }
}

bool IsFile(const std::string &path)
{
    if (!IsPathExist(path))
    {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

bool IsFolder(const std::string &path)
{
    if (!IsPathExist(path))
    {
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

void draw_objects(const cv::Mat &image,
                  cv::Mat &res,
                  const std::vector<det::Object> &objs,
                  const std::vector<std::string> &CLASS_NAMES,
                  const std::vector<std::vector<unsigned int>> &COLORS)
{
    res = image.clone();
    for (auto &obj : objs)
    {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}

float iou(cv::Rect bb_test, cv::Rect bb_gt) {
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return in / un;
}

void nms(std::vector<Object>& res, float nms_thresh) {
    std::map<float, std::vector<Object>> m;
    for (const auto& obj: res) {
        if (m.count(obj.label) == 0) {
            m.emplace(obj.label, std::vector<Object>());
        }
        m[obj.label].push_back(obj);
    }
    auto cmp = [](const Object& a, const Object& b) {
        return a.prob > b.prob;
    };
    res.clear();
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.rect, dets[n].rect) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}