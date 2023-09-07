# Stardust-tensorrt
使用tensorrt在Jetson Orin Nano部署yolov5和yolov8

## Yolov5
### 模型准备
```
// 将pytorch模型转化为onnx
git clone -b v7.0 https://github.com/ultralytics/yolov5.git
cd yolov5/
// 下载pytorch模型（或使用自己训练的模型）
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
// 转换为onnx
python detect.py --weights yolov5s.pt --include onnx
// 在Jetson平台上将onnx模型转换为tensorrt
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov5s.onnx \
--saveEngine=yolov5s.engine
```

### ROS2节点配置

#### 修改cfg文件
```
yolo:
  ros__parameters:
    model_name: yolov5
    engine_path: {PATH-TO-ENGINE}
    conf_thres: 0.25
    nms_thres:  0.65
    class_num:  80
```

## Yolov8
### 模型准备
与yolov5不同，yolov8项目可将nms模块纳入到onnx模型中，故须在该步骤设置iou-thres与conf-thres两个参数
```
// 将pytorch模型转化为onnx
git clone https://github.com/triple-Mu/YOLOv8-TensorRT.git
cd YOLOv8-TensorRT/
// 下载pytorch模型（或使用自己训练的模型）
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
// 转换为onnx
python3 export-det.py \
--weights yolov8s.pt \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0
// 在Jetson平台上将onnx模型转换为tensorrt
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s.onnx \
--saveEngine=yolov8s.engine
```

### ROS2节点配置

#### 修改cfg文件
```
yolo:
  ros__parameters:
    model_name: yolov8
    engine_path: {PATH-TO-ENGINE}
```

## ROS2节点
现在支持onnxruntime和tensorrt两种推理框架，根据cfg文件中加载的模型文件的后缀来动态选择推理框架
```
// 编译节点
colcon build
// 若当前平台不支持tensorrt，可以使用以下命令
colcon build --cmake-args "-DUSE_TENSORRT=OFF"
// 运行
ros2 launch stardust_tensorrt test.py
```