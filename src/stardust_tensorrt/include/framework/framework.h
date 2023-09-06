#ifndef JETSON_DETECT_FRAMEWORK_H
#define JETSON_DETECT_FRAMEWORK_H

#include <vector>
#include "common.h"

class BaseFramework
{
public:
    BaseFramework() {}
    virtual ~BaseFramework() {}
    virtual void forward(const cv::Mat &image, std::vector<void *> &output) = 0;
};

#endif