#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "vo.h"
using namespace visual_odometry;


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
  fs.appendFeaturesFromImage(sample_image);
  for(int age: fs.ages){
    assert(age == 0);
  }
  for(int strength: fs.strengths){
    assert(strength >= FAST_THRESHOLD);
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
  std::vector<bool> okLocations = findUnmovedPoints(points1, points2, .5);
  for(int i = 0; i < 35; i++){
    assert(okLocations[i] == ((i % 5) && (i % 7)));
  }
}

void test_circularMatching() {
  std::vector<cv::Point2f> pl0, pr0, pl1, pr1, pret;
  cv::Mat iL0 = cv::imread("run1/left/frame000001.png", cv::IMREAD_GRAYSCALE);
  cv::Mat iR0 = cv::imread("run1/right/frame000001.png", cv::IMREAD_GRAYSCALE);
  cv::Mat iL1 = cv::imread("run1/left/frame000004.png", cv::IMREAD_GRAYSCALE);
  cv::Mat iR1 = cv::imread("run1/right/frame000004.png", cv::IMREAD_GRAYSCALE);
  FeatureSet fs;
  fs.appendFeaturesFromImage(iL0);
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
  assert(ok >= 30); // Got 30 with original branch, want more if possible
}

void test_deleteFeaturesWithFailureStatus() {
  std::vector<cv::Point2f> pl0, pr0, pl1, pr1, pret;
  cv::Mat iL0 = cv::imread("run1/left/frame000001.png", cv::IMREAD_GRAYSCALE);
  cv::Mat iR0 = cv::imread("run1/right/frame000001.png", cv::IMREAD_GRAYSCALE);
  cv::Mat iL1 = cv::imread("run1/left/frame000004.png", cv::IMREAD_GRAYSCALE);
  cv::Mat iR1 = cv::imread("run1/right/frame000004.png", cv::IMREAD_GRAYSCALE);
  FeatureSet fs;
  fs.appendFeaturesFromImage(iL0);
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
  deleteFeaturesWithFailureStatus(pl0, pr0, pl1, pr1, pret, fs, status);
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
}
// void test_cameraToWorld() {
//   std::vector<cv::Mat> world_points(3, cv::Mat(0, 27));
//   for(unsigned int i = -1; i <= 1; i++){
//     for(unsigned int j = -1; j <= 1; j++){
//       for(unsigned int k = -1; k <= 1; k++){
//         world_points[0].push_back(i);
//       }
//     }
//   }
// }

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
    // run_tests();
    int N_FRAMES = std::stoi(argv[1]);

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
    VisualOdometry vo(projMatrl, projMatrr);
    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
    for(int i = 0; i < N_FRAMES; i++) {
      dbg(i);
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
      std::pair<cv::Mat, cv::Mat> out =  vo.stereo_callback(cur_img_l, cur_img_r);
      cv::Mat translation = out.first;
      cv::Mat rotation = out.second;
      assert(isRotationMatrix(rotation));
      visual_odometry::integrateOdometryStereo(frame_pose, rotation, translation);
      double x = frame_pose.col(3).at<double>(0);
      double y = frame_pose.col(3).at<double>(1);
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