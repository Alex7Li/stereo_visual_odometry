#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vo.h>
#include <iostream>


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
    
    // Read images & store images in correct format
    const cv::Mat prev_img_l =  cv::imread(prev_img_l_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat prev_img_r =  cv::imread(prev_img_r_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat cur_img_l =  cv::imread(cur_img_l_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat cur_img_r =  cv::imread(cur_img_r_path, cv::IMREAD_GRAYSCALE);

    //cv::imwrite("./disparity.png",  (cv::Mat) disp_image);

}
