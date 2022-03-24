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
  const cv::Mat sample_image =  cv::imread("run1images/left/frame000000.png", cv::IMREAD_GRAYSCALE);
  assert(fs.size() == 0);
  fs.appendFeaturesFromImage(sample_image);
  for(int age: fs.ages){
    assert(age == 0);
  }
  for(int strength: fs.strengths){
    assert(strength >= FAST_THRESHOLD);
    assert(strength <= 100);
  }
  assert(fs.size() >= 100); /* Should detect quite a few points, I got 125 */
  fs.filterByBucketLocationInternal(sample_image, 1, 1, 0, 77); /* Put it all in one bucket */
  assert(fs.size() == 77);
}
void test_featureset_filter() {
  FeatureSet fs;
  const cv::Mat sample_image =  cv::imread("run1images/left/frame000000.png", cv::IMREAD_GRAYSCALE);
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

void test_rotationMatrixToEulerAngles(){
  // Maybe
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
  std::cout << "ALL TESTS PASS" << std::endl;
  // assert(false);
  // std::cout << "NEVERMIND ASSERTS WERE JUST DISABLED" << std::endl;
}

// /usr/bin/clang++ -fdiagnostics-color=always -g /home/alex/git/stereo_visual_odometry/src/main.cpp -o /home/alex/git/stereo_visual_odometry/src/main `pkg-config opencv --cflags --libs` -v
int main(int argc, char** argv) {
    run_tests();
    std::string folderName = "run1images";

    float fx= 220.44908;
    float fy= 220.44908;
    float cx= 222.01352;
    float cy= 146.41498;
    float bf= -10.97633;

    // Disparity image data stucture
    cv::Mat_<int16_t> disparity16;

    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);

    VisualOdometry vo(projMatrl, projMatrr);
    for(int i = 0; i < 100; i++) {
        std::stringstream lFileName;
        std::stringstream rFileName;
        lFileName << folderName << "/left/frame" << std::setw(6) << std::setfill('0') << i << ".png";
        rFileName << folderName << "/right/frame" << std::setw(6) << std::setfill('0') << i << ".png";
        const cv::Mat cur_img_l =  cv::imread(lFileName.str(), cv::IMREAD_GRAYSCALE);
        const cv::Mat cur_img_r =  cv::imread(rFileName.str(), cv::IMREAD_GRAYSCALE);
        // std::cout << cur_img_l.channels() << " " << cur_img_l.rows << " " << cur_img_l.cols << std::endl;
        // std::cout << cur_img_l << std::endl;
        // std::cout << cur_img_l.channels() << " " << cur_img_r.rows << " " << cur_img_r.cols << std::endl;
        vo.stereo_callback(cur_img_l, cur_img_r);
    }
    
    //cv::imwrite("./disparity.png",  (cv::Mat) disp_image);
    return 0;
}
