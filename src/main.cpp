#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "vo.h"
#include "pose_estimation_node.h"
#define N_FRAMES 128


bool isRotationMatrix(const cv::Mat &R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
     
    return  norm(I, shouldBeIdentity) < 1e-6;
}

// /usr/bin/clang++ -fdiagnostics-color=always -g /home/alex/git/stereo_visual_odometry/src/main.cpp -o /home/alex/git/stereo_visual_odometry/src/main `pkg-config opencv --cflags --libs` -v
int main(int argc, char** argv) {
    std::string folderName = "run1";
    std::ifstream ground_truth;
    ground_truth.open(folderName + "/gt.csv");
    std::stringstream buffer;
    buffer << ground_truth.rdbuf();
    ground_truth.close();

    std::string lastline;
    std::getline(buffer, lastline);
    std::getline(buffer, lastline, ',');
    std::ofstream resultbuffer;
    resultbuffer.open (folderName + "/result.csv");
    resultbuffer << "x," << "y," << "gtx," << "gty" << "\n";

    float left_P[3][4] = {{322.11376, 0.0, 327.47336, 0.0},
                          {0.0, 322.11376, 176.33722, 0.0},
                          {0.0, 0.0, 1.0, 0.0}};
    float right_P[3][4] = {{322.11376, 0.0, 327.47336, -22.5428},
                           {0.0, 322.11376, 176.33722, 0.0},
                           {0.0, 0.0, 1.0, 0.0}};
    cv::Mat projMatrl(3, 4, CV_32F, left_P);
    cv::Mat projMatrr(3, 4, CV_32F, right_P);
    PoseEstimator vo(projMatrl, projMatrr);
    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
    double x =0;
    double y =0;
    for(int i = 0; i < N_FRAMES; i++) {
      std::string xstr, ystr, dxstr, dystr;
      // apparently we just doesn't read the first col of the csv
      std::getline(buffer, xstr, ',');
      std::getline(buffer, ystr, ',');
      std::getline(buffer, dxstr, ',');
      std::getline(buffer, dystr, ',');
      double gtx = std::stod(xstr);
      double gty = std::stod(ystr);
      std::stringstream lFileName;
      std::stringstream rFileName;
      lFileName << folderName << "/left/frame" << std::setw(6) << std::setfill('0') << i << ".png";
      rFileName << folderName << "/right/frame" << std::setw(6) << std::setfill('0') << i << ".png";
      const cv::Mat cur_img_l =  cv::imread(lFileName.str(), cv::IMREAD_GRAYSCALE);
      const cv::Mat cur_img_r =  cv::imread(rFileName.str(), cv::IMREAD_GRAYSCALE);
      // if (i % 2 !=0) continue;
      // std::pair<cv::Mat, cv::Mat> out = 
      std::pair<double, double> out = vo.stereo_callback(cur_img_l, cur_img_r);
      // cv::Mat translation = out.first;
      // cv::Mat rotation = out.second;
      double dx = out.first;
      double dy = out.second;
      x += dx;
      y += dy;
      // If there was any update
      // assert(isRotationMatrix(rotation));

      // cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);
      // // Don't perform an update if the output is unusually large, indicates a error elsewhere.
      // if (abs(rotation_euler[1]) < 0.1 && abs(rotation_euler[0]) < 0.1 &&
      //     abs(rotation_euler[2]) < 0.1) {
      //   integrateOdometryStereo(i, frame_pose, rotation, translation);
      // }else{
      //   std::cout << "Rotation too big, skipping update\n";
      // }
      // double x = frame_pose.col(3).at<double>(0);
      // double y = frame_pose.col(3).at<double>(1);
      std::cout << "\nx: " << x << " (" << gtx << ")\ty:" <<  y << " (" << gty << ") ";
      resultbuffer << x << "," << y << "," << gtx << "," << gty << "\n";
    }
    std::cout << std::endl;
    resultbuffer.close();
    
    //cv::imwrite("./disparity.png",  (cv::Mat) disp_image);
    return 0;
}
    // Original
    // float fx= 220.44908;
    // float fy= 220.44908;
    // float cx= 222.01352;
    // float cy= 146.41498;
    // float bf= -10.97633;
    // cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    // cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);