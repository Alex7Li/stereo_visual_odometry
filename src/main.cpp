#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "vo.h"
using namespace visual_odometry;

bool isRotationMatrix(const cv::Mat &R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
     
    return  norm(I, shouldBeIdentity) < 1e-6;
}

std::pair<cv::Mat, cv::Mat> readImages(const std::string folderName, int i) {
    std::stringstream lFileName;
    std::stringstream rFileName;
    int zeros = 4;
    std::string end = ".jpg";
    if (folderName == "run1"){
      zeros = 6;
      end = ".png";
    }
    if(folderName == "cfe_cameras"){
      lFileName << folderName << "/SeqID_"<< i << "-CamId_0-post-rectification-stereo_app.png";
      rFileName << folderName << "/SeqID_"<< i << "-CamId_1-post-rectification-stereo_app.png";
    } else{
      lFileName << folderName << "/left/frame" << std::setw(zeros) << std::setfill('0') << i << end;
      rFileName << folderName << "/right/frame" << std::setw(zeros) << std::setfill('0') << i << end;
    }
    const cv::Mat cur_img_l =  cv::imread(lFileName.str(), cv::IMREAD_GRAYSCALE);
    const cv::Mat cur_img_r =  cv::imread(rFileName.str(), cv::IMREAD_GRAYSCALE);
    //  It's already rectified!
   return std::make_pair(cur_img_l, cur_img_r);
    // float left_P[3][3] = {{361.49914, 0., 345.32559},
    //                           {0., 361.49914, 174.00476},
    //                           {0.0, 0.0, 1.0}};
    // // Left Distortion Paramters
    // float left_D[5] = {
    //     0.008135, -0.006633, -0.000483, -0.000344, 0.000000,
    // };
    // // Right Perspective Matrix
    // float right_P[3][3] = {{361.49914, 0., 345.32559},
    //                              {0., 361.49914, 174.00476},
    //                              {0.0, 0.0, 1.0}};
    // // Right Distortion Parameters
    // float right_D[5] = {
    //     0.008283, -0.005682, -0.000120, 0.000139, 0.000000,
    // };
    // // Right Intrinsic Matrix
    // float right_K[3][3] = {
    //     {313.54715, 0., 325.29634}, {0., 316.71185, 187.56471}, {0., 0., 1.}};
    // // Left Intrinsic Matrix
    // float left_K[3][3] = {
    //     {312.60837, 0., 341.50514}, {0., 315.74639, 160.55705}, {0., 0., 1.}};
    // // Right Rectification Matrix
    // float right_R[3][3] = {{0.9998113, -0.00949429, -0.01694758},
    //                              {0.00945161, 0.99995196, -0.00259632},
    //                              {0.01697142, 0.00243565, 0.99985301}};
    // // Left Rectification Matrix
    // float left_R[3][3] = {{0.9997124, -0.0154679, -0.01832675},
    //                             {0.01551397, 0.99987683, 0.0023741},
    //                             {0.01828777, -0.00265774, 0.99982923}};
    // // Read images & store images in correct format
    // cv::Mat left_rect, right_rect;
    // const cv::Mat cur_img_l =  cv::imread(lFileName.str(), cv::IMREAD_GRAYSCALE);
    // const cv::Mat cur_img_r =  cv::imread(rFileName.str(), cv::IMREAD_GRAYSCALE);
    // cv::Mat map_left_x, map_left_y, map_right_x, map_right_y;
    // cv::Mat left_camera_matrix(3,3,CV_32F), left_distortion(1,5,CV_32F);
    // cv::Mat left_rotation(3,3,CV_32F), left_projection(3,3,CV_32F);
    // cv::Mat right_camera_matrix(3,3,CV_32F), right_distortion(1,5, CV_32F);
    // cv::Mat right_rotation(3,3,CV_32F), right_projection(3,3,CV_32F);
    // std::memcpy(left_distortion.data, left_D,
    //             sizeof(float) * 5);
    // std::memcpy(left_camera_matrix.data, left_K,
    //             sizeof(float) * 3 * 3);
    // std::memcpy(left_rotation.data, left_R,
    //             sizeof(float) * 3 * 3);
    // std::memcpy(left_projection.data, left_P,
    //             sizeof(float) * 3 * 3);
    // std::memcpy(right_distortion.data, right_D,
    //             sizeof(float) * 5);
    // std::memcpy(right_camera_matrix.data, right_K,
    //             sizeof(float) * 3 * 3);
    // std::memcpy(right_rotation.data, right_R,
    //             sizeof(float) * 3 * 3);
    // std::memcpy(right_projection.data, right_P,
    //             sizeof(float) * 3 * 3);
    // cv::initUndistortRectifyMap(
    //   left_camera_matrix, left_distortion, left_rotation, left_projection,
    //   cur_img_l.size(), CV_32FC1, map_left_x, map_left_y);
    // cv::initUndistortRectifyMap(
    //     right_camera_matrix, right_distortion, right_rotation, right_projection,
    //     cur_img_r.size(), CV_32FC1, map_right_x, map_right_y);
    // cv::remap(cur_img_l, left_rect, map_left_x, map_left_y,
    //           cv::INTER_LINEAR);
    // cv::remap(cur_img_r, right_rect, map_right_x, map_right_y,
    //           cv::INTER_LINEAR);
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

cv::Mat_<unsigned char> makeEmptyImage(int n_rows, int n_cols) {
    cv::Mat_<unsigned char> image(n_rows, n_cols, CV_8UC1);
    for(int i = 0; i < n_rows; i++){
      for(int j = 0; j < n_cols; j++){
          image[i][j] = 0;
      }
    }
    return image;
}
void addTriangle(cv::Mat_<unsigned char> image, int x, int y, int r) {
    // FAST somehow is bad at detecting most geometric shapes so we do this
    for(int i = -r; i <= r; i++) {
      for(int j = 0; j <= r; j++) {
        if(abs(i) <= r - abs(j)){
          image[i + y][j + x] = (unsigned char)120;
        }
      }
    }
    // a little hole for fast to see a feature
    image[y][x] = 0;
}

void test_featureset() {
  FeatureSet fs;
  assert(fs.size() == 0);

  const cv::Mat sample_image = makeEmptyImage(300, 200);
  for(int i = 0; i <= 10; i++){
    addTriangle(sample_image, 20, (i + 1) * 20, 8);
  }
  fs.appendFeaturesFromImage(sample_image, 1);
  // displayPoints(sample_image, fs.points);
  // cv::imshow("vis ", sample_image);  
  // cv::waitKey(0.01);
  dbg(fs.size());
  for(int age: fs.ages) {
    assert(age == 0);
  }
  for(int strength: fs.strengths){
    // assert(strength >= FAST_THRESHOLD);
    assert(strength <= 128);
  }
  dbga(fs.strengths);
  dbga(fs.points);
  assert(fs.size() == 11);
  fs.filterByBucketLocationInternal(sample_image, 1, 1, 0, 7); /* Put it all in one bucket */
  assert(fs.size() == 7);
}
void test_featureset_filter() {
  FeatureSet fs;
  const cv::Mat sample_image = makeEmptyImage(300, 300);
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
  const cv::Mat iL0 = makeEmptyImage(300, 300);
  const cv::Mat iR0 = makeEmptyImage(300, 300);
  const cv::Mat iL1 = makeEmptyImage(300, 300);
  const cv::Mat iR1 = makeEmptyImage(300, 300);
  for(int i = 0; i <= 10; i++){
    for(int j = 0; j <= 10; j++){
      addTriangle(iL0, (j + 1) * 20, (i + 1) * 20, 8);
      addTriangle(iL1, (j + 1) * 20 + 1, (i + 1) * 20, 8);
      addTriangle(iR0, (j + 1) * 20, (i + 1) * 20 + 1, 8);
      addTriangle(iR1, (j + 1) * 20 + 1, (i + 1) * 20 + 1, 8);
    }
  }
  cv::imshow("image ", iL0);
  cv::waitKey(0.01);
  VisualOdometry vo;
  std::vector<cv::Point2f> pl0, pr0, pl1, pr1, pret;
  vo.stereo_callback(iL0,iR0);
  FeatureSet fs;
  fs.appendFeaturesFromImage(iL0, FAST_THRESHOLD);
  unsigned int n_points = fs.points.size();
  // Check that it doesn't crash on boundary conditions
  vo.circularMatching(iL1, iR1, pl0, pr0, pl1, pr1, fs);
  pl0.push_back(fs.points[0]);
  vo.circularMatching(iL1, iR1, pl0, pr0, pl1, pr1, fs);
  // run
  pl0 = fs.points;
  vo.circularMatching(iL1, iR1, pl0, pr0, pl1, pr1, fs);
  n_points = fs.points.size();
  dbg(n_points);
  assert(pl0.size() == n_points);
  assert(pl1.size() == n_points);
  assert(pr0.size() == n_points);
  assert(pr1.size() == n_points);
  assert(n_points == 121);
}

void test_cameraToWorld() {
  float camera_matrix[3][3] = {{1.0, 0., 0.0},
                        {0., 1.0, 0.0},
                        {0.0, 0.0, 1.0}};
  cv::Mat projMatrl(3, 3, CV_32F, camera_matrix);
  int n_points = 27;
  cv::Mat_<float> world_point_x(cv::Size(1, n_points));
  cv::Mat_<float> world_point_y(cv::Size(1, n_points));
  cv::Mat_<float> world_point_z(cv::Size(1, n_points));
  std::vector<cv::Point2f> camera_points(n_points);
  int ind = 0;
  for(int i = -1; i <= 1; i++){
    for(int j = -1; j <= 1; j++){
      for(int k = 5; k <= 7; k++){
        // there should be a 90 degree rotation along the xy plane
        world_point_x(0, ind) = -j;
        world_point_y(0, ind) = i;
        world_point_z(0, ind) = k;
        // increase k by 1, so there should be a shift in the z direction by 1.
        camera_points[ind] = cv::Point2f(float(i)/ (k + 1), float(j)/ (k + 1));
        ind += 1;
      } 
    }
  }
  std::vector<cv::Mat> world_point_channels;
  world_point_channels.push_back(world_point_x);
  world_point_channels.push_back(world_point_y);
  world_point_channels.push_back(world_point_z);
  cv::Mat world_points;
  cv::merge(world_point_channels, world_points);
  cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
  auto result = cameraToWorld(projMatrl,
      camera_points, world_points, rotation, translation);
  int n_inliers = result.first.size().height;
  assert(abs(translation.at<double>(0)) < 1e-6);
  assert(abs(translation.at<double>(1)) < 1e-6);
  assert(abs(translation.at<double>(2) - 1) < 1e-6);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      if (i + j == 4) {
        assert(abs(rotation.at<double>(i,j) - 1) < 1e-8);
      } else if(i == 1 and j == 0) {
        assert(abs(rotation.at<double>(i,j) + 1) < 1e-8);
      } else if(i == 0 and j == 1) {
        assert(abs(rotation.at<double>(i,j) - 1) < 1e-8);
      } else {
        assert(abs(rotation.at<double>(i,j)) < 1e-8);
      }
    }
  }
  assert(result.second);
  assert(n_inliers == n_points);
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
  test_cameraToWorld();
  std::cout << "ALL TESTS PASS" << std::endl;
  // assert(false);
  // std::cout << "NEVERMIND ASSERTS WERE JUST DISABLED" << std::endl;
}

// /usr/bin/clang++ -fdiagnostics-color=always -g /home/alex/git/stereo_visual_odometry/src/main.cpp -o /home/alex/git/stereo_visual_odometry/src/main `pkg-config opencv --cflags --libs` -v
int main(int argc, char** argv) {
  // extrinsics with kalibr
    int N_FRAMES = 2;
    if(argc == 1){
      std::cout << "No N_FRAMES given, defaulting to 2" << std::endl;
    } else {
      N_FRAMES = std::stoi(argv[1]);
      if(N_FRAMES == 0){
        run_tests();
      }
    }
    std::string folderName = "cfe_cameras";
    bool has_ground_truth = true;
    if(folderName == "rand_feats" || folderName == "cfe_cameras"){
      has_ground_truth = false;
    }
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

    // float left_P[3][4] = {{361.49914, 0., 345.32559, 0.0},
    //                           {0., 361.49914, 174.00476, 0.0},
    //                           {0.0, 0.0, 1.0, 0.0}};
    // float right_P[3][4] = {{361.49914, 0., 345.32559, -22.5428},
    //                           {0., 361.49914, 174.00476, 0.0},
    //                           {0.0, 0.0, 1.0, 0.0}};
    // Original Proejction Left/Right Intrinsic Matrix
    float left_P[3][4] = {{322.11376, 0.0, 327.47336, 0.0},
                          {0.0, 322.11376, 176.33722, 0.0},
                          {0.0, 0.0, 1.0, 0.0}};
    float right_P[3][4] = {{322.11376, 0.0, 327.47336, -22.5428},
                           {0.0, 322.11376, 176.33722, 0.0},
                           {0.0, 0.0, 1.0, 0.0}};
    cv::Mat projMatrl(3, 4, CV_32F, left_P);
    cv::Mat projMatrr(3, 4, CV_32F, right_P);
    VisualOdometry vo;
    vo.initalize_projection_matricies(projMatrl, projMatrr);
    /**
     * @brief The current estimated pose of the robot.
     */
    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
    for(int i = 0; i < N_FRAMES; i++) {
      std::pair<cv::Mat, cv::Mat> images = readImages(folderName, i);
      if(images.first.size().height == 0){
        continue;
      }
      assert(images.second.size().height != 0);

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
      std::pair<bool, cv::Mat> out =  vo.stereo_callback(images.first, images.second);
      // bool success = out.first;
      cv::Mat transform = out.second;
      frame_pose = frame_pose * transform;
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
