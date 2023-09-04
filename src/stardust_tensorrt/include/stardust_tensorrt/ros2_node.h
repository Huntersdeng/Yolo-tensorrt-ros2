#ifndef STARDUST_TENSORRT_ROS2_NODE_H
#define STARDUST_TENSORRT_ROS2_NODE_H

#include <string>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include "stardust_tensorrt/model.h"
#include "stardust_tensorrt/yolov8.h"
#include "stardust_tensorrt/yolov5.h"

class DetectionNode: public rclcpp::Node {
public:
    DetectionNode();
    ~DetectionNode();
    void initialize_publishers();
    void initialize_subscribers();
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg_rgb, const sensor_msgs::msg::Image::ConstSharedPtr msg_depth);
private:
    // yolov8
    std::string m_engine_file_path;
    std::shared_ptr<Model> m_model;

    // ros2
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_depth;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_color;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> syncpolicy;
    typedef message_filters::Synchronizer<syncpolicy> Sync;
    std::shared_ptr<Sync> sync;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_detection;
};

#endif //STARDUST_TENSORRT_ROS2_NODE_H