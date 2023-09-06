#include "ros2_node.h"
#include <ament_index_cpp/get_package_share_directory.hpp>

using namespace det;

DetectionNode::DetectionNode() : Node("YOLO", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
{
    std::string model_name, class_txt;
    float conf_thres, nms_thres;
    int class_num, width, height;
    this->get_parameter("model_name", model_name);
    this->get_parameter("engine_path", m_engine_file_path);
    this->get_parameter_or("class_txt", class_txt, std::string("coco.txt"));
    this->get_parameter_or("conf_thres", conf_thres, 0.25f);
    this->get_parameter_or("nms_thres", nms_thres, 0.65f);
    this->get_parameter_or("class_num", class_num, 80);
    this->get_parameter_or("width", width, 640);
    this->get_parameter_or("height", height, 640);

    class_txt = ament_index_cpp::get_package_share_directory("stardust_tensorrt") + "/config/" + class_txt;
    read_class_name(class_txt, m_class_names_);

    RCLCPP_INFO(this->get_logger(), "Initializing %s from %s", model_name.c_str(), m_engine_file_path.c_str());
    RCLCPP_INFO(this->get_logger(), "Reading class names from %s", class_txt.c_str());
    RCLCPP_INFO(this->get_logger(), "Conf-thres: %f, NMS-thres: %f, class-num: %d, width: %d, height: %d", conf_thres, nms_thres, class_num, width, height);

    if (model_name == "yolov5")
        m_model = std::make_shared<YOLOv5>(m_engine_file_path, conf_thres, nms_thres, class_num, cv::Size(width, height));
    else if (model_name == "yolov8")
        m_model = std::make_shared<YOLOv8>(m_engine_file_path);
    else
        m_model = std::make_shared<YOLOv5>(m_engine_file_path);
    initialize_subscribers();
    initialize_publishers();
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
}

void DetectionNode::initialize_publishers()
{
    std::string output_topic;
    this->get_parameter_or("output_topic", output_topic, std::string("trash_detection/detection"));
    RCLCPP_INFO(this->get_logger(), "Publish topic: %s", output_topic.c_str());
    pub_detection = this->create_publisher<sensor_msgs::msg::Image>(output_topic, 1);
}

void DetectionNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg_rgb, const sensor_msgs::msg::Image::ConstSharedPtr msg_depth)
{
    cv::Mat img_raw = cv_bridge::toCvCopy(msg_rgb, sensor_msgs::image_encodings::RGB8)->image;
    std::vector<det::Object> objs;
    auto start = std::chrono::system_clock::now();
    m_model->detect(img_raw, objs);
    auto end = std::chrono::system_clock::now();

    cv::Mat res;
    draw_objects(img_raw, res, objs, m_class_names_, COLORS);
    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    RCLCPP_INFO(this->get_logger(), "cost %2.4lf ms\n", tc);

    pub_detection->publish(*(cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", res).toImageMsg()));
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}