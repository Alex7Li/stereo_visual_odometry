/****************************************************************
 *
 * @file 		vo.h
 *
 * @brief 		The Visual Odometry class being used for
 translation. The math can be found in Haidar Jamal's Thesis:
 *https://www.ri.cmu.edu/publications/localization-for-lunar-micro-rovers/
 *
 * @version 	1.0
 * @date 		02/09/2022
 *
 * @authors 	Ben Kolligs, Alex Li
 * @author 		Carnegie Mellon University, Planetary Robotics Lab
 *
 ****************************************************************/
#include <ctype.h>
#include <math.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "Core"
#include "Dense"

extern "C" {
#include "cfe_error.h"
#include "common_types.h"
#include "pe_events.h"
}

namespace visual_odometry {
/**
 * @brief Number of buckets along each axis of the image.
 * In total, there will be BUCKETS_PER_AXIS * BUCKETS_PER_AXIS
 * buckets.
 */
// TODO @Future change to different bucket sizes per axis
int BUCKETS_PER_AXIS = 10
/**
 * @brief Maximum number of features per bucket
 */
int FEATURES_PER_BUCKET = 1;
/**
 * @brief Ignore all features that have been around but not detected
 * for this many frames.
 */
int AGE_THRESHOLD = 10;


// TODO @Alex7Li comment what the Bucket class does
class Bucket {
public:
  int id;
  int max_size;

  // TODO @Alex7Li comment what this is meant to store
  FeatureSet features;

  // TODO @Alex7Li name this
  Bucket(int);
  ~Bucket();

  /** //TODO @Alex7Li comment this
   * @brief DESCRIPTION 
   *
   * @param point DESCRIPTION
   * @param age DESCRIPTION
   */
  void add_feature(const cv::Point2f point, const int age);

  int size();
};

// TODO @Alex7Li comment what the FeatureSet class does
class FeatureSet {
public:
  
  // TODO @Alex7Li comment what this is meant to store
  std::vector<cv::Point2f> points;
  // TODO @Alex7Li comment what this is meant to store
  std::vector<int> ages;
  int size() { return points.size(); }
  /**
   * @brief Updates the feature set to only include a subset of the
   * original features which give a good spread throughout the image.
   *
   * @param image only use for getting dimension of the image
   *
   * @param bucket_size bucket size in pixel is bucket_size*bucket_size
   *
   * @param features_per_bucket: number of selected features per bucket
   */

  void filterByBucketLocation(const cv::Mat &image, const int bucket_size,
                              const int features_per_bucket);

  /**
   * @brief Apply a feature detection algorithm over the image to generate new
   * features, and all all such features into this feature set.
   *
   * @param image //  TODO @Alex7Li Description
   */
  void appendFeaturesFromImage(const cv::Mat &image);
};

class VisualOdometry {
private:
  /* number of frames seen so far. */
  int frame_id = 0;
  /* Projection matrices for the left and right cameras */
  cv::Mat leftCameraProjection_, rightCameraProjection_;

  /* Images at current and next time step */
  cv::Mat imageRightT0_, imageLeftT0_;
  cv::Mat imageRightT1_, imageLeftT1_;

  /* Initial pose variables */
  cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);

  // TODO @Alex7Li What is this
  cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);

  /* Set of features currently tracked. */
  FeatureSet currentVOFeatures;

  //  TODO @Alex7Li Comment this
  void run();

public:
  /**
   * @brief Construct a new Visual Odometry object
   *
   * @param ProjMatrl Left camera projection matrix
   *
   * @param projMatrr Right camera projection matrix
   */
  VisualOdometry(const cv::Mat ProjMatrl, const cv::Mat projMatrr);

  ~VisualOdometry();

  /**
   * @brief Process one time step of camera imagery and
   * publish the result.
   *
   * @param image_left The left image from stereo camera
   *
   * @param image_right The right image from stereo camera
   */
  void stereo_callback(const cv::Mat &image_left, const cv::Mat &image_right);
};

/**
 * @brief Use the FAST feature detector to accumulate the features in image into
 * points.
 *
 * @param image The image we're detecting.
 *
 * @return A vector with the locations of all newly detected features.
 * //TODO @Alex7Li this does not return anything??
 */
void featureDetectionFast(const cv::Mat image);

/**
 * @brief Remove all points from the 4 point vectors that are out of frame
 * or have a status of 0 //TODO @Alex7Li also old??
 *
 * @param points[0..4] vectors of points to update based on status.
 * SHOULD ALL BE THE SAME SIZE AND REPRESENT //TODO @Alex7Li what do they represent
 *
 * @param ages Current ages of each of the points
 *
 * @param status_all a vector with 1 If the point is valid, and 0 if it should be discarded.
 */
// TODO @Alex7Li rename this
void deleteUnmatchFeaturesCircle(
    std::vector<cv::Point2f> &points0, std::vector<cv::Point2f> &points1,
    std::vector<cv::Point2f> &points2, std::vector<cv::Point2f> &points3,
    std::vector<cv::Point2f> &points4, std::vector<int> &ages
    const std::vector<uchar> &status_all);

/**
 * @brief Perform circular matching on  TODO @Alex7Li on what
 * Detect points not found in both cameras for both the previous and
 * current frame and remove them.
 */
// TODO @Alex7Li separate out deleteUnmatchFeaturesCircle
void circularMatching(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1,
                      cv::Mat img_r_1, std::vector<cv::Point2f> &points_l_0,
                      std::vector<cv::Point2f> &points_r_0,
                      std::vector<cv::Point2f> &points_l_1,
                      std::vector<cv::Point2f> &points_r_1,
                      std::vector<cv::Point2f> &points_l_0_return,
                      FeatureSet &current_features);
/**
 * @brief Compute the next pose from the current one
 * given the rotation and translation in the frame.
 * Essentially a multiplicationof homogeneous transforms.
 * 
 * @param frame_pose The original position of the robot, will be modified.
 *
 * @param rotation The rotation to go through.
 *
 * @param translation_stereo The translation to go through.
 */
void integrateOdometryStereo(cv::Mat &frame_pose,
                             const cv::Mat &rotation,
                             const cv::Mat &translation_stereo);
/**
 * @brief Compute the three euler angles for a given rotation matrix.
 *
 * @param R A rotation matrix
 *
 * @return cv::Vec3f (x, y, z) euler angles for R.
 */
cv::Vec3f rotationMatrixToEulerAngles(const cv::Mat &R);
/**
 * @brief Given two vectors of points, find the locations where they
 * differ.
 * 
 * @param points_[1..2] The vectors of points to compare.
 *
 * @param threshold The distance at which to consider the points moved.
 *
 * @return a vector v where v[i] is true iff |points_1[i] - points_2[i]| <= threshold
 */
std::vector<bool> findUnmovedPoints(const std::vector<cv::Point2f> &points_1,
                     const std::vector<cv::Point2f> &points_2,
                     const int threshold);
/**
 * @brief Update a vector of points, removing all of the points[i]
 * in which status[i] is false.
 * 
 * @param points The vector of points to update
 *
 * @param status A vector indicating which points to remove.
 */
void removeInvalidPoints(std::vector<cv::Point2f> &points,
                         const std::vector<bool> &status);
/**
 * @brief Compute the translation and rotation that needs to occur
 * to obtain the given world points from the camera input points
 * 
 * @param cameraProjection Camera projection matrix
 *
 * @param cameraPoints Points from the perspective of the camera
 *
 * @param worldPoints Same points in the real world
 *
 * @param rotation Matrix to store the estimated rotation output in.
 *
 * @param translation Matrix to store the estimated translation in.
 */
void cameraToWorld(cv::Mat &cameraProjection,
                         std::vector<cv::Point2f> &cameraPoints,
                         cv::Mat &worldPoints, cv::Mat &rotation,
                         cv::Mat &translation);
/**
 * // TODO @Alex7Li 
 * @brief <<DESCRIBE>>
 * Calls many of the above functions in a pipeline.
 * a->b->c
 * 
 * Input: 4 images and the set of currently tracked features, as
 * well as references to 4 vectors of points (by reference).
 * 
 * @return: vectors of features shared between the 4 images.
 */
void matchingFeatures(const cv::Mat &imageLeftT0, const cv::Mat &imageRightT0,
                      const cv::Mat &imageLeftT1, const cv::Mat &imageRightT1,
                      FeatureSet &currentVOFeatures,
                      std::vector<cv::Point2f> &pointsLeftT0,
                      std::vector<cv::Point2f> &pointsRightT0,
                      std::vector<cv::Point2f> &pointsLeftT1,
                      std::vector<cv::Point2f> &pointsRightT1);
} // namespace visual_odometry
