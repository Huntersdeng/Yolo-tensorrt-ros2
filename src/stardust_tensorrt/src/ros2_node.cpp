#include "ros2_node.h"
#include <ament_index_cpp/get_package_share_directory.hpp>

using namespace det;

DetectionNode::DetectionNode() : Node("YOLO", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
{
    std::string model_name, class_txt;
    float conf_thres, nms_thres;
    int class_num, width, height;
    this->get_parameter("model_name", model_name);
    this->get_parameter("model_type", m_model_type_);
    this->get_parameter("engine_path", m_engine_file_path);
    this->get_parameter_or("class_txt", class_txt, std::string("coco.txt"));
    this->get_parameter_or("conf_thres", conf_thres, 0.25f);
    this->get_parameter_or("nms_thres", nms_thres, 0.65f);
    this->get_parameter_or("width", width, 640);
    this->get_parameter_or("height", height, 640);

    class_txt = ament_index_cpp::get_package_share_directory("stardust_tensorrt") + "/config/" + class_txt;
    read_class_name(class_txt, m_class_names_);
    class_num = m_class_names_.size();

    RCLCPP_INFO(this->get_logger(), "Initializing %s-%s from %s", model_name.c_str(), m_model_type_.c_str(), m_engine_file_path.c_str());
    RCLCPP_INFO(this->get_logger(), "Reading class names from %s", class_txt.c_str());
    RCLCPP_INFO(this->get_logger(), "Conf-thres: %f, NMS-thres: %f, class-num: %d, width: %d, height: %d", conf_thres, nms_thres, class_num, width, height);

    if (model_name == "yolov5") {
        if (m_model_type_ == "det") {
            m_model = std::make_shared<YOLOv5>(m_engine_file_path, cv::Size(width, height), conf_thres, nms_thres, class_num);
        } else {
            RCLCPP_ERROR(this->get_logger(), "%s-%s is not implemented", model_name.c_str(), m_model_type_.c_str());
            assert(false);
        }
    }     
    else if (model_name == "yolov8") {
        if (m_model_type_ == "det") {
            m_model = std::make_shared<YOLOv8>(m_engine_file_path);
        } else if (m_model_type_ == "seg") {
            m_model = std::make_shared<YOLOv8Seg>(m_engine_file_path, cv::Size(width, height), conf_thres, nms_thres, cv::Size(160, 160), 32);
        } else {
            RCLCPP_ERROR(this->get_logger(), "%s-%s is not implemented", model_name.c_str(), m_model_type_.c_str());
            assert(false);
        }
    }
    else {
        RCLCPP_ERROR(this->get_logger(), "%s-%s is not implemented", model_name.c_str(), m_model_type_.c_str());
        assert(false);
    }
    initialize_subscribers();
    initialize_publishers();

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
}

DetectionNode::~DetectionNode() {}

void DetectionNode::initialize_subscribers()
{
    std::string rgb_topic, depth_topic;
    this->get_parameter_or("rgb_topic", rgb_topic, std::string("/color/color_raw"));
    this->get_parameter_or("depth_topic", depth_topic, std::string("/depth/depth_raw"));
    RCLCPP_INFO(this->get_logger(), "Subscribe topic: %s and %s", rgb_topic.c_str(), depth_topic.c_str());
    sub_color.subscribe(this, rgb_topic);
    sub_depth.subscribe(this, depth_topic);
    sync.reset(new Sync(syncpolicy(1), sub_color, sub_depth));
    sync->registerCallback(std::bind(&DetectionNode::image_callback, this, std::placeholders::_1,
                                     std::placeholders::_2));

    sub_cam_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/depth/camera_info", 10,
        std::bind(&DetectionNode::cb_get_cam_intrinsic, this, std::placeholders::_1));
}

void DetectionNode::initialize_publishers()
{
    std::string output_topic;
    this->get_parameter_or("output_topic", output_topic, std::string("trash_detection/detection"));
    RCLCPP_INFO(this->get_logger(), "Publish topic: %s", output_topic.c_str());
    pub_detection = this->create_publisher<sensor_msgs::msg::Image>(output_topic, 1);

    pub_trash_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("trash_detection/trash_cloud", 1);
}

void DetectionNode::initialize_tf(std::string target_frame_id)
{
    if (is_tf_initialized) return;
    geometry_msgs::msg::TransformStamped msg_transform;
    try {
        msg_transform = tf_buffer_->lookupTransform(base_frame_id, target_frame_id, tf2::TimePointZero);
    } catch (tf2::TransformException& ex) {
        RCLCPP_ERROR_STREAM(this->get_logger(),
                            "Transform error of sensor data: " << ex.what() << ", quitting callback");
        return;
    }
    pcl_ros::transformAsMatrix(msg_transform, transform_cam);
    is_tf_initialized = true;
}

void DetectionNode::cb_get_cam_intrinsic(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    /*
        @Describe:camera1 内参
    */
    cam_intrinsic.fxParam = msg->k[0];
    cam_intrinsic.cxParam = msg->k[2];
    cam_intrinsic.fyParam = msg->k[4];
    cam_intrinsic.cyParam = msg->k[5];
}

void DetectionNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg_rgb, const sensor_msgs::msg::Image::ConstSharedPtr msg_depth)
{
    // yolo推理
    cv::Mat img_raw = cv_bridge::toCvCopy(msg_rgb, sensor_msgs::image_encodings::RGB8)->image;
    std::vector<det::Object> objs;
    auto start = std::chrono::system_clock::now();
    m_model->detect(img_raw, objs);
    auto end = std::chrono::system_clock::now();

    // 绘制目标检测结果并发布
    cv::Mat res;
    if (m_model_type_ == "seg") {
        draw_objects_masks(img_raw, res, objs, m_class_names_, COLORS, MASK_COLORS);
    } else {
        draw_objects(img_raw, res, objs, m_class_names_, COLORS);
    }
    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    RCLCPP_INFO(this->get_logger(), "cost %2.4lf ms\n", tc);

    pub_detection->publish(*(cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", res).toImageMsg()));

    // // 获取目标点云
    // initialize_tf(msg_depth->header.frame_id);

    // std::vector<pcl::PointCloud<pcl::PointXYZ>> clouds;
    // get_objects_cloud(msg_depth, objs, clouds);

    // sensor_msgs::msg::PointCloud2 ros_cloud;
    // pcl::PointCloud<pcl::PointXYZ> tmp_cloud;
    // for (const auto& cloud : clouds) {
    //     tmp_cloud += cloud;
    // }
    // toROSMsg(tmp_cloud, ros_cloud);
    // ros_cloud.header.frame_id = base_frame_id;
    // ros_cloud.header.stamp = rclcpp::Clock().now();
    // pub_trash_cloud->publish(ros_cloud);
}

void DetectionNode::get_objects_cloud(const sensor_msgs::msg::Image::ConstSharedPtr msg_depth, const std::vector<Object> &objs, 
                                      std::vector<pcl::PointCloud<pcl::PointXYZ>> &clouds) {
    /*
        @Describe: depth 转 点云
    */
    for (const auto& obj : objs) {
        pcl::PointCloud<pcl::PointXYZ> camera_cloud, base_cloud;
        get_object_cloud(msg_depth, obj, camera_cloud);
        pcl::transformPointCloud(camera_cloud, base_cloud, transform_cam);
        clouds.push_back(std::move(base_cloud));
    }
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr down_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    // pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> vg;
    // vg.setInputCloud(cloud_raw);
    // vg.setLeafSize(leafSize, leafSize, leafSize);
    // vg.filter(*cloud);
}

void DetectionNode::get_object_cloud(const sensor_msgs::msg::Image::ConstSharedPtr msg_depth, const Object &obj, 
                                      pcl::PointCloud<pcl::PointXYZ> &cloud) {
    int x = obj.rect.x;
    int y = obj.rect.y;
    int w = obj.rect.width;
    int h = obj.rect.height;
    cloud.clear();
    cloud.height = h;
    cloud.width = w;
    cloud.resize(cloud.height * cloud.width);
    int32_t index = 0;
    uint16_t* pData = (uint16_t*)&msg_depth->data[0];
    float tmpDepthValue = 0;
    auto pt_iter = cloud.begin();
    for (int j = y; j < y + h; ++j) {
        for (int i = x; i < x + w; ++i) {
            index = j * msg_depth->width + i;
            tmpDepthValue = (pData[index] * 1.0) / 1000.0;

            pcl::PointXYZ& o = *pt_iter++;
            if (tmpDepthValue > 0) {
                o.z = tmpDepthValue;
                o.x = ((float)(i - cam_intrinsic.cxParam) * o.z) / cam_intrinsic.fxParam;
                o.y = ((float)(j - cam_intrinsic.cyParam) * o.z) / cam_intrinsic.fyParam;
            } else {
                o.z = 0.0f;
                o.x = 0.0f;
                o.y = 0.0f;
            }
        }
    }
}  

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}