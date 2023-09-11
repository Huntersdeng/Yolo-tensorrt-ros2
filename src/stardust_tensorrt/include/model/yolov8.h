#ifndef JETSON_DETECT_YOLOV8_H
#define JETSON_DETECT_YOLOV8_H
#include "model/model.h"
#include "fstream"

class YOLOv8 : public Model
{
public:
    explicit YOLOv8(const std::string &engine_file_path);
    ~YOLOv8();

protected:
    void postprocess(const std::vector<void*> output, std::vector<det::Object> &objs);

};

#endif // JETSON_DETECT_YOLOV8_H