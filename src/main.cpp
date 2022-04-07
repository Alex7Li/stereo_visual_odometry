#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "vo.h"
using namespace visual_odometry;

std::pair<cv::Mat, cv::Mat> readImages(const std::string folderName, int i) {
    std::stringstream lFileName;
    std::stringstream rFileName;
    int zeros = 6;
    std::string end = ".png";
    if (folderName == "rand_feats"){
      zeros = 4;
      end = ".jpg";
    }
    lFileName << folderName << "/left/frame" << std::setw(zeros) << std::setfill('0') << i << end;
    rFileName << folderName << "/right/frame" << std::setw(zeros) << std::setfill('0') << i << end;
    float left_P[3][3] = {{361.49914, 0., 345.32559},
                              {0., 361.49914, 174.00476},
                              {0.0, 0.0, 1.0}};
    // Left Distortion Paramters
    float left_D[5] = {
        0.008135, -0.006633, -0.000483, -0.000344, 0.000000,
    };
    // Right Perspective Matrix
    float right_P[3][3] = {{361.49914, 0., 345.32559},
                                 {0., 361.49914, 174.00476},
                                 {0.0, 0.0, 1.0}};
    // Right Distortion Parameters
    float right_D[5] = {
        0.008283, -0.005682, -0.000120, 0.000139, 0.000000,
    };
    // Right Intrinsic Matrix
    float right_K[3][3] = {
        {313.54715, 0., 325.29634}, {0., 316.71185, 187.56471}, {0., 0., 1.}};
    // Right Rectification Matrix
    float right_R[3][3] = {{0.9998113, -0.00949429, -0.01694758},
                                 {0.00945161, 0.99995196, -0.00259632},
                                 {0.01697142, 0.00243565, 0.99985301}};
    // Left Intrinsic Matrix
    float left_K[3][3] = {
        {312.60837, 0., 341.50514}, {0., 315.74639, 160.55705}, {0., 0., 1.}};
    // Left Rectification Matrix
    float left_R[3][3] = {{0.9997124, -0.0154679, -0.01832675},
                                {0.01551397, 0.99987683, 0.0023741},
                                {0.01828777, -0.00265774, 0.99982923}};
    // Read images & store images in correct format
    cv::Mat left_rect, right_rect;
    const cv::Mat cur_img_l =  cv::imread(lFileName.str(), cv::IMREAD_GRAYSCALE);
    const cv::Mat cur_img_r =  cv::imread(rFileName.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat map_left_x, map_left_y, map_right_x, map_right_y;
    cv::Mat left_camera_matrix(3,3,CV_32F), left_distortion(1,5,CV_32F);
    cv::Mat left_rotation(3,3,CV_32F), left_projection(3,3,CV_32F);
    cv::Mat right_camera_matrix(3,3,CV_32F), right_distortion(1,5, CV_32F);
    cv::Mat right_rotation(3,3,CV_32F), right_projection(3,3,CV_32F);
    std::memcpy(left_distortion.data, left_D,
                sizeof(float) * 5);
    std::memcpy(left_camera_matrix.data, left_K,
                sizeof(float) * 3 * 3);
    std::memcpy(left_rotation.data, left_R,
                sizeof(float) * 3 * 3);
    std::memcpy(left_projection.data, left_P,
                sizeof(float) * 3 * 3);
    std::memcpy(right_distortion.data, right_D,
                sizeof(float) * 5);
    std::memcpy(right_camera_matrix.data, right_K,
                sizeof(float) * 3 * 3);
    std::memcpy(right_rotation.data, right_R,
                sizeof(float) * 3 * 3);
    std::memcpy(right_projection.data, right_P,
                sizeof(float) * 3 * 3);
    cv::initUndistortRectifyMap(
      left_camera_matrix, left_distortion, left_rotation, left_projection,
      cur_img_l.size(), CV_32FC1, map_left_x, map_left_y);
    cv::initUndistortRectifyMap(
        right_camera_matrix, right_distortion, right_rotation, right_projection,
        cur_img_r.size(), CV_32FC1, map_right_x, map_right_y);
    cv::remap(cur_img_l, left_rect, map_left_x, map_left_y,
              cv::INTER_LINEAR);
    cv::remap(cur_img_r, right_rect, map_right_x, map_right_y,
              cv::INTER_LINEAR);
   // It's already rectified!
   return std::make_pair(cur_img_l, cur_img_r);
   // It's not rectified :(
  //  return std::make_pair(left_rect, right_rect);
}


void test_bucket_empty() {
  Bucket b(0);
  std::vector<cv::Point2f> points(5, {1, 1});
  std::vector<int> ages = {6, 2, 3, 4, 5};
  std::vector<int> strengths = {60, 20, 30, 40, 50};
  for(int i = 0; i < 5; i++){
    b.add_feature(points[i], ages[i], strengths[i]);
  }
  assert(b.max_size == 0);
  assert(b.features.size() == 0);
}

void test_bucket_nonempty() {
  Bucket b(3);
  std::vector<cv::Point2f> points(5, {1, 1});
  std::vector<int> ages = {6, 2, 3, 4, 5};
  std::vector<int> strengths = {60, 20, 30, 40, 50};
  for(int i = 0; i < 5; i++){
    b.add_feature(points[i], ages[i], strengths[i]);
  }
  assert(b.max_size == 3);
  assert(b.features.size() == 3);
  assert(b.features.ages[0] == 6);
  assert(b.features.ages[1] == 4);
  assert(b.features.ages[2] == 5);
  assert(b.features.strengths[0] == 60);
  assert(b.features.strengths[1] == 40);
  assert(b.features.strengths[2] == 50);
}

void test_featureset() {
  FeatureSet fs;
  const cv::Mat sample_image =  cv::imread("run1/left/frame000000.png", cv::IMREAD_GRAYSCALE);
  assert(fs.size() == 0);
  fs.appendFeaturesFromImage(sample_image, FAST_THRESHOLD);
  for(int age: fs.ages) {
    assert(age == 0);
  }
  for(int strength: fs.strengths){
    // assert(strength >= FAST_THRESHOLD);
    assert(strength <= 100);
  }
  assert(fs.size() >= 10); /* Should detect quite a few points, I got 125 */
  fs.filterByBucketLocationInternal(sample_image, 1, 1, 0, 7); /* Put it all in one bucket */
  assert(fs.size() == 7);
}
void test_featureset_filter() {
  FeatureSet fs;
  const cv::Mat sample_image =  cv::imread("run1/left/frame000000.png", cv::IMREAD_GRAYSCALE);
  const int image_rows = sample_image.rows;
  const int image_cols = sample_image.cols;
  const int bucket_height = (image_rows +  1) / 2;
  // Note rows and cols appear 'flipped' because of how images are read in opencv
  for(int i = 0; i < 15; i++){
    fs.points.push_back({ image_cols - 1.f, image_rows - 1.f});
    fs.ages.push_back(0);
    fs.strengths.push_back(40);
  }
  for(int i = 0; i < 10; i++){
    fs.points.push_back({image_cols - 1.f, 0.f});
    fs.ages.push_back(0);
    fs.strengths.push_back(40);
  }
  for(int i = 0; i < 5; i++){
    fs.points.push_back({0.f, float(bucket_height)});
    fs.ages.push_back(0);
    fs.strengths.push_back(40);
  }

  assert(fs.size() == 30);
  fs.filterByBucketLocationInternal(sample_image, 2, 2, 0, 11); 
  FeatureSet fs_copy = fs;
  assert(fs.size() == 26);
  fs.filterByBucketLocationInternal(sample_image, 2, 1, 0, 11); 
  assert(fs.size() == 21);
  fs_copy.filterByBucketLocationInternal(sample_image, 1, 2, 0, 11); 
  assert(fs_copy.size() == 16);
}

void test_findUnmovedPoints(){
  std::vector<cv::Point2f> points1;
  std::vector<cv::Point2f> points2;
  for(int i = 0; i < 35; i++){
    points1.push_back({ float(i), float(i)});
    points2.push_back({ float(i) + !(i % 5), float(i) + !(i % 7)});
  }
  std::vector<bool> okLocations = findClosePoints(points1, points2, .5);
  for(int i = 0; i < 35; i++){
    assert(okLocations[i] == ((i % 5) && (i % 7)));
  }
}

void test_circularMatching() {
  std::vector<cv::Point2f> pl0, pr0, pl1, pr1, pret;
  auto t0 = readImages("run1/", 0);
  auto t1 = readImages("run1/", 1);
  cv::Mat iL0 = t0.first;
  cv::Mat iR0 = t0.second;
  cv::Mat iL1 = t1.first;
  cv::Mat iR1 = t1.second;
  FeatureSet fs;
  fs.appendFeaturesFromImage(iL0, FAST_THRESHOLD);
  unsigned int n_points = fs.points.size();
  // Check that it doesn't crash on boundary conditions
  std::vector<bool> status = circularMatching(iL0, iR0, iL1, iR1, pl0, pr0, pl1, pr1, pret);
  assert(status.size() == 0);
  pl0.push_back(fs.points[0]);
  status = circularMatching(iL0, iR0, iL1, iR1, pl0, pr0, pl1, pr1, pret);
  assert(status.size() == 1);
  // run
  pl0 = fs.points;
  status = circularMatching(iL0, iR0, iL1, iR1, pl0, pr0, pl1, pr1, pret);
  assert(status.size() == n_points);
  assert(pl0.size() == n_points);
  assert(pl1.size() == n_points);
  assert(pr0.size() == n_points);
  assert(pr1.size() == n_points);
  assert(pret.size() == n_points);
  int ok = 0;
  for(unsigned int i = 0; i < status.size(); i++){
    if(status[i]){
      ok++;
    }
  }
  // assert(ok >= 30); // Got 30 with original branch, want more if possible
}

void test_deleteFeaturesWithFailureStatus() {
  std::vector<cv::Point2f> pl0, pr0, pl1, pr1, pret;
  auto t0 = readImages("run1/", 0);
  auto t1 = readImages("run1/", 1);
  cv::Mat iL0 = t0.first;
  cv::Mat iR0 = t0.second;
  cv::Mat iL1 = t1.first;
  cv::Mat iR1 = t1.second;
  FeatureSet fs;
  fs.appendFeaturesFromImage(iL0, FAST_THRESHOLD);
  pl0 = fs.points;
  std::vector<bool> status = circularMatching(iL0, iR0, iL1, iR1, pl0, pr0, pl1, pr1, pret);
  std::vector<int> okStrengths, okAges;
  std::vector<cv::Point2f> okPoints;
  for(unsigned int i = 0; i < status.size(); i++){
      fs.ages[i] = i;
      if(status[i]){
        okPoints.push_back(fs.points[i]);
        okStrengths.push_back(fs.strengths[i]);
        okAges.push_back(fs.ages[i]);
      }
  }
  deleteFeaturesAndPointsWithFailureStatus(pl0, pr0, pl1, pr1, pret, fs, status);
  assert(fs.strengths.size() == okPoints.size());
  assert(fs.points.size() == okPoints.size());
  assert(fs.ages.size() == okPoints.size());
  for(unsigned int i = 0; i < okPoints.size(); i++){
    assert(okPoints[i] == fs.points[i]);
    assert(okStrengths[i] == fs.strengths[i]);
    assert(okAges[i] == fs.ages[i]);
  }
  assert(pl0.size() == okPoints.size());
  assert(pl1.size() == okPoints.size());
  assert(pr0.size() == okPoints.size());
  assert(pr1.size() == okPoints.size());
  assert(pret.size() == okPoints.size());
  // visualize
  cv::Mat rgbimg = cv::imread("run1/left/frame000000.png", cv::IMREAD_COLOR);
  for(int r = 0; r < t0.first.rows; r++) {
    for(int c = 0; c < t0.first.cols; c++) {
      uint8_t intensity = iL0.at<uint8_t>(r, c);
      cv::Vec3b color = {intensity, intensity, intensity};
      rgbimg.at<cv::Vec3b>(r, c) = color;
    }
  }
  for(cv::Point2f p : pret){
      cv::Vec3b& color = rgbimg.at<cv::Vec3b>(p.y, p.x);
      color[0] = 255;
      color[1] = 0;
      color[2] = 0;
      rgbimg.at<cv::Vec3b>(p.y, p.x) = color;
  }
  const std::string filename = "run1/out.png";
  bool success = cv::imwrite(filename, rgbimg);
  if(!success) {
    dbgstr("failed to write image");
  }else{
    dbgstr("wrote image");
  }
}
void test_cameraToWorld() {
  cv::Mat world_points(cv::Size(27, 3), CV_64F);
  for(unsigned int i = -1; i <= 1; i++){
    for(unsigned int j = -1; j <= 1; j++){
      for(unsigned int k = -1; k <= 1; k++){
        int ind = i + 1 + ( j + 1) * 3 + (k + 1) * 9;
        world_points.at<double>(ind, 0)= (double)i;
        world_points.at<double>(ind, 1)= (double)j;
        world_points.at<double>(ind, 2) = (double)k;
      }
    }
  }

  cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
  // cv::Mat inliers = cameraToWorld(leftCameraProjection_,
  //     pointsLeftT1, world_points, rotation, translation);
}
void test_goes_forward() {

}

void run_tests() {
  std::cout << "TEST BUCKET EMPTY" << std::endl;
  test_bucket_empty();
  std::cout << "TEST BUCKET NONEMPTY" << std::endl;
  test_bucket_nonempty();
  std::cout << "TEST FEATURE SET" << std::endl;
  test_featureset();
  std::cout << "TEST FEATURE SET EDGE" << std::endl;
  test_featureset_filter();
  std::cout << "TEST FIND UNMOVED POINTS" << std::endl;
  test_findUnmovedPoints();
  std::cout << "TEST CIRCULAR MATCHING" << std::endl;
  test_circularMatching();
  std::cout << "TEST DELETE FEATURES" << std::endl;
  test_deleteFeaturesWithFailureStatus();
  // std::cout << "TEST CAMERA TO WORLD" << std::endl;
  // test_cameraToWorld();
  // TODO: Cameratoworld, matchingFeatures maybe (mostly tested)? 

  std::cout << "ALL TESTS PASS" << std::endl;
  // assert(false);
  // std::cout << "NEVERMIND ASSERTS WERE JUST DISABLED" << std::endl;
}

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
  // extrinsics with kalibr
    // run_tests();
    int N_FRAMES = std::stoi(argv[1]);
    std::string folderName = "run1";//"rand_feats";
    bool has_ground_truth = true;
    bool useRot = true;
    // std::string folderName = "run1";
    std::stringstream buffer;
    if(has_ground_truth) {
      std::ifstream ground_truth;
      ground_truth.open(folderName + "/gt.csv");
      buffer << ground_truth.rdbuf();
      ground_truth.close();
      std::string lastline;
      std::getline(buffer, lastline);
      std::getline(buffer, lastline, ',');
    }

    std::ofstream resultbuffer;
    resultbuffer.open (folderName + "/result.csv");
    resultbuffer << "x,y,z,gtx,gty" << "\n";

    float left_P[3][4] = {{322.11376, 0.0, 327.47336, 0.0},
                          {0.0, 322.11376, 176.33722, 0.0},
                          {0.0, 0.0, 1.0, 0.0}};
    float right_P[3][4] = {{322.11376, 0.0, 327.47336, -22.5428},
                           {0.0, 322.11376, 176.33722, 0.0},
                           {0.0, 0.0, 1.0, 0.0}};
    cv::Mat projMatrl(3, 4, CV_32F, left_P);
    cv::Mat projMatrr(3, 4, CV_32F, right_P);
    VisualOdometry vo(projMatrl, projMatrr);
    cv::Mat frame_pose = vo.frame_pose;
    for(int i = 0; i < N_FRAMES; i++) {
      std::cout << "Frame " << i << ": ";
      double gtx = 0;
      double gty = 0;
      if (has_ground_truth) {
        std::string xstr, ystr, dxstr, dystr;
        // apparently we just doesn't read the first col of the csv
        std::getline(buffer, xstr, ',');
        std::getline(buffer, ystr, ',');
        std::getline(buffer, dxstr, ',');
        std::getline(buffer, dystr, ',');
        gtx = std::stod(xstr);
        gty = std::stod(ystr);
      }
      std::pair<cv::Mat, cv::Mat> images = readImages(folderName, i);
      std::pair<cv::Mat, cv::Mat> out =  vo.stereo_callback(images.first, images.second);
      cv::Mat translation = out.first;
      cv::Mat rotation = out.second;
      assert(isRotationMatrix(rotation));
      if (useRot){
        visual_odometry::integrateOdometryStereo(frame_pose, rotation, translation);
      } else {
        // This still goes in the wrong direction. Why?
        frame_pose.col(3).at<double>(0) += translation.at<double>(0);
        frame_pose.col(3).at<double>(1) += translation.at<double>(1);
        frame_pose.col(3).at<double>(2) += translation.at<double>(2);
      }
      double x = frame_pose.col(3).at<double>(0);
      double y = frame_pose.col(3).at<double>(1);
      double z = frame_pose.col(3).at<double>(2);
      resultbuffer << x << "," << y << "," << z << "," << gtx << "," << gty << "\n";
    }
    std::cout << std::endl;
    resultbuffer.close();
    // idk what this is
    //cv::imwrite("./disparity.png",  (cv::Mat) disp_image);
    return 0;
}
