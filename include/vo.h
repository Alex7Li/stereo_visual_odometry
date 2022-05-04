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
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#define dbg(x) std::cerr << " >>> " << #x << " = " << x << std::endl;
#define dbga(a) std::cerr << " >>> " << #a << " = "; for(unsigned int xtw = 0; xtw < a.size(); xtw++) std::cerr << a[xtw] << " "; std::cerr << std::endl;
#define dbgstr(a) std::cerr << " >>> " << a << std::endl;
/************
 * ROS ONLY *
 ************/
// #include "ros/ros.h"
// #include "sensor_msgs/Image.h"
// #include "std_msgs/Int32MultiArray.h"
// #include "nav_msgs/Odometry.h"
// #include "geometry_msgs/Quaternion.h"
// #include <tf/transform_broadcaster.h>
// #include <message_filters/subscriber.h>
// #include <message_filters/synchronizer.h>
// #include <message_filters/sync_policies/approximate_time.h>
// #include <cv_bridge/cv_bridge.h>

// #include <eigen3/Eigen/Dense>
// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Geometry> 
/* END ROS ONLY */

/************
 * CFS ONLY *
 ************/
/*
#include "Core"
#include "Dense"

extern "C" {
#include "cfe_error.h"
#include "common_types.h"
#include "pe_events.h"
}
*/
/* END CFS ONLY */

namespace visual_odometry {

/**
 * @brief Skip this many top rows of buckets,
 * since they often correspond to points on the horizon
 * which are hard to triangulate.
 */
const int BUCKET_START_ROW = 4;

/**
 * @brief Number of buckets along each axis of the image.
 * In total, there will be BUCKETS_ALONG_WIDTH * BUCKETS_ALONG_HEIGHT
 * buckets. Note the image size is 366x644.
 */
const int BUCKETS_ALONG_HEIGHT = 92;
const int BUCKETS_ALONG_WIDTH  = 160;
/**
 * @brief Maximum number of features per bucket.
 */
const int FEATURES_PER_BUCKET = 1;

/**
 * @brief Minimum number of features required after
 * circular matching before vo fails.
 */
const int FEATURES_THRESHOLD = 20;

/**
 * @brief Minimum number of features before
 * circular matching matching before we lower the
 * threshold to find more.
 */
const int PRE_MATCHING_FEATURE_THRESHOLD = 100;

/**
 * @brief Ignore all features that have been around but not detected
 * for this many frames.
 */
const int AGE_THRESHOLD = 20;

/**
 * @brief Minimum confidence for the robot to report a feature
 * detection in the FAST algorithm.
 */
const int FAST_THRESHOLD = 20;

/**
 * @brief Distance in pixels for that the projection of triangulated
 * points must have onto the image to be considered an inlier when
 * using RANSAC.
 */
const float RANSAC_REPROJECTION_ERROR = 6;

/**
 * @brief Maximum number of iterations for RANSAC.
 */
const int RANSAC_ITERATIONS = 500;

/**
 * @brief The minimum eigenvalue of the spacial gradient matrix calculated
 * by calcOpitcalFlowPyrLK.
 */
const double OPTICAL_FLOW_MIN_EIG_THRESHOLD = 0.01;

/**
 * @brief Threshold for a feature to be considered successfully matched by
 * circular matching. The circularly matched feature is required to be at
 * least this close to the original point in pixels.
 */
const double CIRCULAR_MATCHING_SUCCESS_THRESHOLD = .15;

/**
 * @brief A set of locations for image features and their ages.
 */
class FeatureSet {
   public:
    /**
     * @brief The points stored in this set of features.
     */
    std::vector<cv::Point2f> points;
    /**
     * @brief A parallel set to points; ages[i] contains
     * the number of iterations that points[i] has been around.
     */
    std::vector<int> ages;

    /**
     * @brief A parallel set to points; strengths[i] contains
     * the clarity of the features reported by the feature detector.
     */
    std::vector<int> strengths;

    /**
     * @brief Return the size of the feature set. Note that
     * points.size() == ages.size()
     *
     * @return size: number of points in the set.
     */
    int size() {
        return points.size();
    }

    /**
     * @brief Updates the feature set to only include a subset of the
     * original features which give a good spread throughout the image.
     *
     * @param image only use for getting dimension of the image.
     */

    void filterByBucketLocation(const cv::Mat &image);

    /* @brief Variant with constant parameters passed in, for testing. */
    void filterByBucketLocationInternal(const cv::Mat &image,
                                        const int buckets_along_height,
                                        const int buckets_along_width,
                                        const int bucket_start_row,
                                        const int features_per_bucket);

    /**
     * @brief Apply a feature detection algorithm over the image to generate new
     * features, and all all such features into this feature set.
     *
     * @param image The image to obtain all points from.
     * @param fast_threshold Threshold for detecting features with FAST.
     */
    void appendFeaturesFromImage(const cv::Mat &image,
                                 const int fast_threshold);
    /**
     * @brief Remove all points from the feature set.
     */
    void clear();
};

/**
 * @brief A class to allow storing a set of at most max_size
 * image features, and remove outdated features to satisfy this
 * constraint.
 **/
class Bucket {
   public:
    int max_size;

    /**
     * @brief The set of features stored in this bucket.
     */
    FeatureSet features;

    Bucket(int max_size);
    ~Bucket();

    /**
     * @brief Rank how good a feature is based on it's current
     * age and strength. Older points that have survived many
     * iterations are desirable, as are ones that were detected
     * strongly.
     */
    int compute_score(const int age, const int strength);
    /**
     * @brief Add a feature to the bucket
     *
     * @param point The location of the feature to add
     * @param age The number of iterations since this feature was detected.
     * @param strength The strength of the detected feature.
     */
    void add_feature(const cv::Point2f point, const int age,
                     const int strength);

    /**
     * @return int The size of the feature set
     */
    int size();
};

class VisualOdometry {
   private:
    /** Number of frames seen so far. */
    int frame_id = 0;
    /** Projection matrices for the left and right cameras. */
    cv::Mat leftCameraProjection_, rightCameraProjection_;
    /** The leftmost 3x3 submatrix of leftCameraProjection_ */
    cv::Mat left_camera_matrix;

    /** Images at current and next time step. */
    cv::Mat imageRightT0_, imageLeftT0_;
    cv::Mat imageRightT1_, imageLeftT1_;

    /** Initial pose variables. */

    /** Set of features currently tracked. */
    FeatureSet currentVOFeatures;

    /**
     * @brief Optical flow calculation hyperparameters:
     * window size and max pyramid level
     */
    cv::Size winSize = cv::Size(7, 7);
    int maxLevel     = 3;

    /**
     * @brief Cached values for the image optical flow pyramids.
     */
    std::vector<cv::Mat> lastLeftPyramid;
    std::vector<cv::Mat> lastRightPyramid;

    /**
     * @brief Result from the last iteration, to be used as
     * a prior belief for this one. last_transform
     * is computed as the inverse of
     * (the translation followed by the rotation).
     */
    cv::Mat_<double> rotation       = cv::Mat_<double>::eye(3, 3);
    cv::Mat_<double> translation    = cv::Mat_<double>::zeros(3, 1);
    cv::Mat_<double> last_transform = cv::Mat_<double>::eye(4, 4);

   public:
    /**
     * @brief Construct a new Visual Odometry object
     *
     * @param ProjMatrl Left camera projection matrix
     *
     * @param projMatrr Right camera projection matrix
     */
    VisualOdometry();


    ~VisualOdometry();

    /**
     * @brief Initialize a Visual Odometry object
     * with the given projection matricies, should
     * be called before anything else.
     *
     * @param leftCameraProjection Left camera projection matrix
     *
     * @param rightCameraProjection Right camera projection matrix
     */
    void initalize_projection_matricies(const cv::Mat leftCameraProjection, const cv::Mat rightCameraProjection);

    /**
     * @brief Process one time step of camera imagery and
     * publish the result.
     *
     * @param image_left The left image from stereo camera
     *
     * @param image_right The right image from stereo camera
     * @return (success, transform):
     * success: A flag, false if we could not produce a useful result (because
     * not enough features were detected or something went wrong and the
     * prediction is very extreme). transform: Transform to move the robot from
     * it's position in the last time step onto the current time step. If the
     * robot's pose is initially the 4x4 matrix of rotation and translation
     * vector
     * frame_pose: A vector of the shape
     * R R R Tx
     * R R R Ty
     * R R R Tz
     * 0 0 0 1
     * The estimted pose after a time step will be frame_pose @ transform
     * where @ denotes matrix multiplication.
     */
    std::pair<bool, cv::Mat_<double>> stereo_callback(
        const cv::Mat &image_left, const cv::Mat &image_right);

    /**
     * @brief Given four images and the set of features used in the last
     * iteration, this method finds new features in the images and appends
     * each of the locations of the features in each image to 4 parallel
     * vectors.
     *
     * Calls many of the above functions in a pipeline.
     * appendFeaturesFromImage -> filterByBucketLocation -> circularMatching ->
     * deleteFeaturesWithFailureStatus -> findClosePoints -> removeInvalidPoints
     *
     * @param image[(Left)|(Right)]T[01]: Images from the left/right cameras at
     * the last/current timestep.
     * @param currentVOFeatures: The set of currently tracked features, stored
     * as a position in the LeftT0 image, will be updated with newly detected
     * features.
     * @param points[(Left)|(Right)][01]: references to 4 empty vectors of
     * points to fill up with feature positions.
     */
    void matchingFeatures(const cv::Mat &imageLeftT0,
                          const cv::Mat &imageRightT0,
                          const cv::Mat &imageLeftT1,
                          const cv::Mat &imageRightT1,
                          FeatureSet &currentVOFeatures,
                          std::vector<cv::Point2f> &pointsLeftT0,
                          std::vector<cv::Point2f> &pointsRightT0,
                          std::vector<cv::Point2f> &pointsLeftT1,
                          std::vector<cv::Point2f> &pointsRightT1);
    /**
     * @brief Perform circular matching on 4 images and
     * detect points not found in both cameras for both the previous and
     * current frame.
     * @param img_[(Left)|(Right)]T1 Images of the same scene taken by different
     * cameras at this timestep.
     * @param points_[(Left)|(Right)]T[01] Features of the scene detected by the
     * cameras.
     * @param current_features The current feature set to consider while
     * performing the circular matching.
     */
    void circularMatching(const cv::Mat &imgLeftT1, const cv::Mat &imgRightT1,
                          std::vector<cv::Point2f> &pointsLeftT0,
                          std::vector<cv::Point2f> &pointsRightT0,
                          std::vector<cv::Point2f> &pointsLeftT1,
                          std::vector<cv::Point2f> &pointsRightT1,
                          FeatureSet &current_features);
};

/**
 * @brief Use the FAST feature detector to accumulate the features in image into
 * points.
 *
 * @param image The image we're detecting.
 * @param fast_threshold The threshold to detect a feature.
 * @param response_strengths: A vector to fill with the response strength of
 * each newly detected feature.
 *
 * @return A vector with the locations of all newly detected features.
 */
std::vector<cv::Point2f> featureDetectionFast(
    const cv::Mat image, const int fast_threshold,
    std::vector<float> &response_strengths);

/**
 * @brief Given a vectors of points,
 * update the vectors by removing elements with a invalid status.
 *
 * @param points_vector vectors of points to update based on status
 *
 * @param status a vector true if point is valid, and 0 if it should be
 * discarded.
 */
void deletePointsWithFailureStatus(std::vector<cv::Point2f> &point_vector,
                                   const std::vector<bool> &isok);
/**
 * @brief Given a set of features, update it by removing elements with a invalid
 * status.
 * @param currentFeatures Current set of features we will need to update.
 *
 * @param status a vector true if point is valid, and 0 if it should be
 * discarded.
 */
void deleteFeaturesWithFailureStatus(FeatureSet &currentFeatures,
                                     const std::vector<bool> &isok);

/**
 * @brief Given two vectors of points, find the locations where they
 * differ.
 *
 * @param points_[1..2] The vectors of points to compare.
 *
 * @param threshold The distance at which to consider the points moved.
 *
 * @return a vector v where v[i] is 1 iff |points_1[i] - points_2[i]| <=
 * threshold, else 0
 */
std::vector<bool> findClosePoints(const std::vector<cv::Point2f> &points_1,
                                  const std::vector<cv::Point2f> &points_2,
                                  float threshold);

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
 *
 * @return (inliers, success)
 * inliers: Indicies of all inliers in the best RANSAC transform
 * success: True if tracking was successful.
 */
std::pair<cv::Mat, bool> cameraToWorld(
    const cv::Mat &cameraProjection,
    const std::vector<cv::Point2f> &cameraPoints, const cv::Mat &worldPoints,
    cv::Mat &rotation, cv::Mat &translation);

/**
 * @brief Compute the inverse transform corresponding to a rotation
 * and tranlation vector.
 *
 * @param rotation The rotation to go through.
 *
 * @param translation_stereo The translation to go through.
 *
 * @return Homogenous matrix containing the inverse transform
 * of translating, then rotating.
 */
cv::Mat getInverseTransform(const cv::Mat &rotation,
                            const cv::Mat &translation_stereo);

}   // namespace visual_odometry
