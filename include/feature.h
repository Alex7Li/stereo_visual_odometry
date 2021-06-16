#ifndef FEATURE_H
#define FEATURE_H

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#if USE_CUDA
  #include <opencv2/cudaoptflow.hpp>
  #include <opencv2/cudaimgproc.hpp>
  #include <opencv2/cudaarithm.hpp>
  #include <opencv2/cudalegacy.hpp>
#endif

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#include <utils.h>

struct FeatureSet {
    std::vector<cv::Point2f>  points;
    std::vector<int>  ages;
    std::vector<int>  strengths;

    int size(){
        return points.size();
    }
    void clear(){
        points.clear();
        ages.clear();
        strengths.clear();
    }
 };

// how many of the top rows of buckets to ignore of the image
const int BUCKET_START_ROW = 2; 

// number of buckets to divide image to. # buckets = BUCKET_DIVISOR*BUCKET_DIVISOR 
const int BUCKET_DIVISOR = 15;

// maximum number of features inside a bucket 
const int FEATURES_PER_BUCKET = 4; // TODO PARAM

// fast feature corner threshold
const int FAST_THRESHOLD = 20;

// feature age threshold
const int AGE_THRESHOLD = 10;

void featureDetectionFast(cv::Mat image, std::vector<cv::Point2f>& points, std::vector<float>& response_strengths);

void featureDetectionGoodFeaturesToTrack(cv::Mat image, std::vector<cv::Point2f>& points);

void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, std::vector<uchar>& status);

void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1,
                          std::vector<cv::Point2f>& points2, std::vector<cv::Point2f>& points3,
                          std::vector<cv::Point2f>& points0_return,
                          std::vector<uchar>& status0, std::vector<uchar>& status1,
                          std::vector<uchar>& status2, std::vector<uchar>& status3,
                          FeatureSet & current_features);

void circularMatching(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1, cv::Mat img_r_1,
                      std::vector<cv::Point2f>& points_l_0, std::vector<cv::Point2f>& points_r_0,
                      std::vector<cv::Point2f>& points_l_1, std::vector<cv::Point2f>& points_r_1,
                      std::vector<cv::Point2f>& points_l_0_return,
                      FeatureSet& current_features);

#if USE_CUDA
  void circularMatching_gpu(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1, cv::Mat img_r_1,
                        std::vector<cv::Point2f>& points_l_0, std::vector<cv::Point2f>& points_r_0,
                        std::vector<cv::Point2f>& points_l_1, std::vector<cv::Point2f>& points_r_1,
                        std::vector<cv::Point2f>& points_l_0_return,
                        FeatureSet& current_features);
#endif

void bucketingFeatures(const cv::Mat& image, FeatureSet& current_features, int bucket_size, int features_per_bucket);

void appendNewFeatures(const cv::Mat& image, FeatureSet& current_features);

void appendNewFeatures(std::vector<cv::Point2f> points_new, FeatureSet& current_features);

void displayTracking(const cv::Mat& imageLeft_t1, 
                     const std::vector<cv::Point2f>&  pointsLeft_t0,
                     const std::vector<cv::Point2f>&  pointsLeft_t1);

void displayPoints(const cv::Mat& image, const std::vector<cv::Point2f>&  points); 

void displayTwoImages(const cv::Mat& image1, const cv::Mat& image2);

#endif
