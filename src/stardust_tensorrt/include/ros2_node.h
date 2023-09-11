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
#include <tf2_ros/transform_broadcaster.h>
#include "pcl_ros/impl/transforms.hpp"

#include <rclcpp/rclcpp.hpp>
#include "model/model.h"
#include "model/yolov8.h"
#include "model/yolov5.h"
#include "model/yolov8_seg.h"

class DetectionNode : public rclcpp::Node
{
public:
    DetectionNode();
    ~DetectionNode();
    void initialize_publishers();
    void initialize_subscribers();
    void initialize_tf(std::string target_frame_id);
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg_rgb, const sensor_msgs::msg::Image::ConstSharedPtr msg_depth);
    void cb_get_cam_intrinsic(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    void get_object_cloud(const sensor_msgs::msg::Image::ConstSharedPtr msg_depth, const det::Object &obj,
                          pcl::PointCloud<pcl::PointXYZ> &cloud);
    void get_objects_cloud(const sensor_msgs::msg::Image::ConstSharedPtr msg_depth, const std::vector<det::Object> &objs,
                           std::vector<pcl::PointCloud<pcl::PointXYZ>> &clouds);

private:
    // yolo
    std::string m_model_type_;
    std::string m_engine_file_path;
    std::shared_ptr<Model> m_model;
    std::vector<std::string> m_class_names_;

    // ros2
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_depth;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_color;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> syncpolicy;
    typedef message_filters::Synchronizer<syncpolicy> Sync;
    std::shared_ptr<Sync> sync;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_cam_info;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    Eigen::Matrix4f transform_cam;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_detection;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_trash_cloud;

    std::string base_frame_id = "base_footprint";
    bool is_tf_initialized = false;

    struct CAM_INTRINSIC
    {
        float fxParam;
        float fyParam;
        float cxParam;
        float cyParam;
    } cam_intrinsic;
};

#endif // STARDUST_TENSORRT_ROS2_NODE_H