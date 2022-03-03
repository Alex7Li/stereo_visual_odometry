#include "vo.h"
#include <stdexcept>

namespace visual_odometry {

cv::Mat rosImage2CvMat(sensor_msgs::ImageConstPtr img) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
            cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
            return cv::Mat();
    }
    return cv_ptr->image;
}

// Entry point
int main(int argc, char **argv)
{

    ros::init(argc, argv, "stereo_vo_node");

    ros::NodeHandle n;

    ros::Rate loop_rate(20);

    std::string filename; //TODO correct the name
    if (!(n.getParam("calib_yaml",filename)))
    {
        std::cerr << "no calib yaml" << std::endl;
        throw;
    }
    // Get default projection matrix parameters from file storage
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(!(fs.isOpened()))
    {
        std::cerr << "cv failed to load yaml" << std::endl;
        throw;
    }
    float fx, fy, cx, cy, bf; // Projection matrix parameters
    fs["fx"] >> fx;
    fs["fy"] >> fy;
    fs["cx"] >> cx;
    fs["cy"] >> cy;
    fs["bf"] >> bf;

    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);

    // initialize VO object
    VisualOdometry stereo_vo(projMatrl, projMatrr);

    // using message_filters to get stereo callback on one topic
    message_filters::Subscriber<sensor_msgs::Image> image1_sub(n, "left/image_rect", 1);
    message_filters::Subscriber<sensor_msgs::Image> image2_sub(n, "right/image_rect", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image1_sub, image2_sub);
    // Register stereo_callback to the synchronizer. When the synchronizer sends a pair of stereo
    // images _1 and _2, we convert them into matricies and run visual odometry.
    sync.registerCallback(boost::bind(&VisualOdometry::stereo_callback, &stereo_vo, 
        boost::bind(rosImage2CvMat, _1), boost::bind(rosImage2CvMat, _2)));

    std::cout << "Stereo VO Node Initialized!" << std::endl;
  
    ros::spin();
    return 0;
}
} // namespace visual_odometry
