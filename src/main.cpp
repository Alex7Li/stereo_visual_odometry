#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "../include/vo.h"

namespace visual_odometry {

int main(int argc, char** argv) {

    if(argc < 5) {
      std::cout << "Too few arguments" << std::endl;
      return -1;
    }
    
    // Disparity image data stucture
    cv::Mat_<int16_t> disparity16;

    // Image paths
    std::string prev_img_l_path = argv[1];
    std::string prev_img_r_path = argv[2];
    std::string cur_img_l_path = argv[1];
    std::string cur_img_r_path = argv[2];
    std::string folderName = "run1images";

    float fx= 282.06762;
    float fy= 282.06762;
    float cx= 290.79884;
    float cy= 182.52132;
    float bf= -16.61097;

    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);

    VisualOdometry vo(projMatrl, projMatrr);
    for(int i = 0; i < 128; i++) {
        std::stringstream lFileName;
        std::stringstream rFileName;
        lFileName << folderName << "/left/" << std::setw(4) << i << ".jpg";
        rFileName << folderName << "/right/" << std::setw(4) << i << ".jpg";
        const cv::Mat cur_img_l =  cv::imread(lFileName.str(), cv::IMREAD_GRAYSCALE);
        const cv::Mat cur_img_r =  cv::imread(rFileName.str(), cv::IMREAD_GRAYSCALE);
        vo.stereo_callback(cur_img_l, cur_img_r);
    }
    
    //cv::imwrite("./disparity.png",  (cv::Mat) disp_image);
    return 0;
}

} // namespace visual_odometry