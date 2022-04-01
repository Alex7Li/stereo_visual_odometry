/****************************************************************
 *
 * @file 		vo.cpp
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
#include "vo.h"

using namespace visual_odometry;
VisualOdometry::VisualOdometry(const cv::Mat leftCameraProjection,
                               const cv::Mat rightCameraProjection) {
  leftCameraProjection_ = leftCameraProjection;
  rightCameraProjection_ = rightCameraProjection;
}

VisualOdometry::~VisualOdometry() {}
std::pair<cv::Mat, cv::Mat> VisualOdometry::stereo_callback(const cv::Mat &imageLeft,
                                      const cv::Mat &imageRight) {
    cv::Mat no_rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat no_translation = cv::Mat::zeros(3, 1, CV_64F);
    // Wait until we have at least two time steps of data
    // to begin predicting the change in pose.
    if (!frame_id) {
      imageLeftT0_ = imageLeft;
      imageRightT0_ = imageRight;
      frame_id++;
      return std::make_pair(no_translation, no_rotation);
    }

    imageLeftT1_ = imageLeft;
    imageRightT1_ = imageRight;

    std::vector<cv::Point2f> pointsLeftT0, pointsRightT0, pointsLeftT1,
        pointsRightT1;
    matchingFeatures(imageLeftT0_, imageRightT0_, imageLeftT1_, imageRightT1_,
                     currentVOFeatures, pointsLeftT0, pointsRightT0,
                     pointsLeftT1, pointsRightT1);

    // Set new images as old images.
    imageLeftT0_ = imageLeftT1_;
    imageRightT0_ = imageRightT1_;
    dbg(currentVOFeatures.size());
    if (currentVOFeatures.size() < FEATURES_THRESHOLD) {
      // There are not enough features to fully determine
      // equations for pose estimation, so presume nothing and exit.
      frame_id++;
      return std::make_pair(no_translation, no_rotation);
    }

    // ---------------------
    // Triangulate 3D Points
    // ---------------------
    cv::Mat world_points_T0, world_homogenous_points_T0;
    cv::triangulatePoints(leftCameraProjection_, rightCameraProjection_,
                          pointsLeftT0, pointsRightT0, world_homogenous_points_T0);
    cv::convertPointsFromHomogeneous(world_homogenous_points_T0.t(), world_points_T0);
    // ---------------------
    // Tracking transfomation
    // ---------------------

    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat inliers = cameraToWorld(leftCameraProjection_,
        pointsLeftT1, world_points_T0, rotation, translation);
    int n_inliers = inliers.size().height;
    // std::vector<bool> is_ok(currentVOFeatures.size());
    // for(int i = 0; i < n_inliers; i++){
    //   is_ok[inliers.at<int>(i)] = true;
    // }
    // dbg(double(n_inliers) / currentVOFeatures.size());
    // deleteFeaturesWithFailureStatus(currentVOFeatures, is_ok);
    double translation_norm = cv::norm(translation);
    cv::Mat rotation_rodrigues;
    cv::Rodrigues(rotation, rotation_rodrigues);  
    double angle = cv::norm(rotation_rodrigues, cv::NORM_L2);
    // dbg(n_inliers);
    if (n_inliers < FEATURES_THRESHOLD) {
      // There are not enough features to fully determine
      // equations for pose estimation, so presume nothing and exit.
      frame_id++;
      // dbgstr("not enough features")
      return std::make_pair(no_translation, no_rotation);
    }

    // ------------------------------------------------
    // Integrating
    // ------------------------------------------------
    if(translation_norm > .1 || abs(angle) > 0.5){// || double(n_inliers) / currentVOFeatures.size() < .4) {
      dbgstr("Translation too suspicious, not updating");
      // dbg(n_inliers);
      // dbg(currentVOFeatures.size());
      // dbg(translation_norm);
      // dbg(abs(angle));
      return std::make_pair(no_translation, no_rotation);
    }
    cv::Mat xyz = frame_pose.col(3).clone();
    cv::Mat R = frame_pose(cv::Rect(0, 0, 3, 3));
    // publish
    // if (true) {
    //     static tf::TransformBroadcaster br;

    //     tf::Transform transform;
    //     transform.setOrigin(tf::Vector3(xyz.at<double>(0), xyz.at<double>(1),
    //                                     xyz.at<double>(2)));
    //     tf::Quaternion q;
    //     tf::Matrix3x3 R_tf(
    //         R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
    //         R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
    //         R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2));
    //     R_tf.getRotation(q);
    //     transform.setRotation(q);
    //     br.sendTransform(tf::StampedTransform(transform, ros::Time::now(),
    //                                           "odom", "camera"));

    //     transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
    //     tf::Quaternion q2(0.5, -0.5, 0.5, -0.5);
    //     transform.setRotation(q2);
    //     br.sendTransform(
    //         tf::StampedTransform(transform, ros::Time::now(), "map",
    //         "odom"));
    // }
    frame_id++;
    return std::make_pair(translation, rotation);
  }

  // --------------------------------
  // https://github.com/hjamal3/stereo_visual_odometry/blob/main/src/feature.cpp
  // --------------------------------

void visual_odometry::deleteFeaturesWithFailureStatus(FeatureSet &currentFeatures,
    const std::vector<bool> &status_all){
  unsigned int indexCorrection = 0;
  for (unsigned int i = 0; i < status_all.size(); i++) {
      if ((status_all.at(i) == 0)) {
        currentFeatures.ages.erase(currentFeatures.ages.begin() + (i - indexCorrection));
        currentFeatures.strengths.erase(currentFeatures.strengths.begin() + (i - indexCorrection));
        currentFeatures.points.erase(currentFeatures.points.begin() + (i - indexCorrection));
        indexCorrection++;
      }
    }
}
void visual_odometry::deleteFeaturesAndPointsWithFailureStatus(
    std::vector<cv::Point2f> &points0, std::vector<cv::Point2f> &points1,
    std::vector<cv::Point2f> &points2, std::vector<cv::Point2f> &points3,
    std::vector<cv::Point2f> &points4, FeatureSet &currentFeatures,
    const std::vector<bool> &status_all) {
  // getting rid of points for which the KLT tracking failed or those who have
  // gone outside the frame
  unsigned int indexCorrection = 0;
  for (unsigned int i = 0; i < status_all.size(); i++) {
      if ((status_all.at(i) == 0)) {
        points0.erase(points0.begin() + (i - indexCorrection));
        points1.erase(points1.begin() + (i - indexCorrection));
        points2.erase(points2.begin() + (i - indexCorrection));
        points3.erase(points3.begin() + (i - indexCorrection));
        points4.erase(points4.begin() + (i - indexCorrection));

        currentFeatures.ages.erase(currentFeatures.ages.begin() + (i - indexCorrection));
        currentFeatures.strengths.erase(currentFeatures.strengths.begin() + (i - indexCorrection));
        currentFeatures.points.erase(currentFeatures.points.begin() + (i - indexCorrection));
        indexCorrection++;
      }
    }
  }

std::vector<bool> visual_odometry::circularMatching(const cv::Mat imgLeft_t0, const cv::Mat imgRight_t0, 
                        const cv::Mat imgLeft_t1, const cv::Mat imgRight_t1,
                        std::vector<cv::Point2f> & pointsLeft_t0,
                        std::vector<cv::Point2f> & pointsRight_t0,
                        std::vector<cv::Point2f> & pointsLeft_t1,
                        std::vector<cv::Point2f> & pointsRight_t1,
                        std::vector<cv::Point2f> & points_0_return) {
    if(pointsLeft_t0.size() == 0){
      std::vector<bool> status_all; 
      return status_all;
    }
    std::vector<float> err;

    cv::Size winSize =
        cv::Size(20, 20); // Lucas-Kanade optical flow window size
    cv::TermCriteria termcrit = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    std::vector<uchar> status0;
    std::vector<uchar> status1;
    std::vector<uchar> status2;
    std::vector<uchar> status3;
    cv::Mat pyramid0;
    cv::Mat pyramid1;
    // cv::buildOpticalFlowPyramid(imgLeft_t0, pyramid0, winSize, 3);
    // cv::buildOpticalFlowPyramid(imgRight_t0, pyramid1, winSize, 3);
    // Sparse iterative version of the Lucas-Kanade optical flow in pyramids.
    
    calcOpticalFlowPyrLK(imgLeft_t0, imgRight_t0, pointsLeft_t0, pointsRight_t0, status0, err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    calcOpticalFlowPyrLK(imgRight_t0, imgRight_t1, pointsRight_t0, pointsRight_t1, status1, err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    calcOpticalFlowPyrLK(imgRight_t1, imgLeft_t1, pointsRight_t1, pointsLeft_t1, status2, err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    calcOpticalFlowPyrLK(imgLeft_t1, imgLeft_t0, pointsLeft_t1, points_0_return, status3, err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    std::vector<bool> status_all(status0.size());
    for(unsigned int i = 0; i < status3.size(); i++) {
      status_all[i] = status0[i] | status1[i] | status2[i] | status3[i];
    }
    return status_all;
}


// --------------------------------
// https://github.com/hjamal3/stereo_visual_odometry/blob/main/src/utils.cpp
// --------------------------------
void visual_odometry::integrateOdometryStereo(cv::Mat &frame_pose, const cv::Mat &rotation,
                              const cv::Mat &translation_stereo) {
  cv::Mat rigid_body_transformation;

  cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);

  cv::hconcat(rotation, translation_stereo, rigid_body_transformation);
  cv::vconcat(rigid_body_transformation, addup, rigid_body_transformation);

  rigid_body_transformation = rigid_body_transformation.inv();
  frame_pose = frame_pose * rigid_body_transformation;
}

// --------------------------------
// https://github.com/hjamal3/stereo_visual_odometry/blob/main/src/visualOdometry.cpp
// --------------------------------

std::vector<bool> visual_odometry::findUnmovedPoints(const std::vector<cv::Point2f> & points_1,
                      const std::vector<cv::Point2f> & points_2,
                      const float threshold) {
  std::vector<bool> status;
  float offset;
  for (unsigned int i = 0; i < points_1.size(); i++) {
    offset = std::max(std::abs(points_1[i].x - points_2[i].x),
                      std::abs(points_1[i].y - points_2[i].y));
    if (offset > threshold) {
      status.push_back(false);
    } else {
      status.push_back(true);
    }
  }
  return status;
}

cv::Mat visual_odometry::cameraToWorld(
    const cv::Mat & cameraProjection,
    const std::vector<cv::Point2f> & cameraPoints, const cv::Mat & worldPoints,
    cv::Mat & rotation, cv::Mat & translation) {
  // Calculate frame to frame transformation
  cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
  // TODO: Why is this defined this way?? Bug??
  cv::Mat intrinsic_matrix =
      (cv::Mat_<float>(3, 3) << cameraProjection.at<float>(0, 0), cameraProjection.at<float>(0, 1), cameraProjection.at<float>(0, 2),
        cameraProjection.at<float>(1, 0), cameraProjection.at<float>(1, 1), cameraProjection.at<float>(1, 2),
        cameraProjection.at<float>(2, 0), cameraProjection.at<float>(2, 1), cameraProjection.at<float>(2, 2));
  int iterationsCount = 500; // number of Ransac iterations.
  float reprojectionError = .5; // maximum allowed distance to consider it an inlier.
  float confidence = 0.999; // RANSAC successful confidence.
  bool useExtrinsicGuess = true;
  int flags = cv::SOLVEPNP_ITERATIVE;

  cv::Mat inliers;
  cv::solvePnPRansac(worldPoints, cameraPoints, intrinsic_matrix, distCoeffs,
                      rvec, translation, useExtrinsicGuess, iterationsCount,
                      reprojectionError, confidence, inliers, flags);

  cv::Rodrigues(rvec, rotation);
  return inliers;
}

void visual_odometry::matchingFeatures(
    const cv::Mat &imageLeft_t0, const cv::Mat &imageRight_t0,
    const cv::Mat &imageLeft_t1, const cv::Mat &imageRight_t1,
    FeatureSet &currentVOFeatures, std::vector<cv::Point2f> &pointsLeftT0,
    std::vector<cv::Point2f> &pointsRightT0,
    std::vector<cv::Point2f> &pointsLeftT1,
    std::vector<cv::Point2f> &pointsRightT1) {
  
  std::vector<cv::Point2f> pointsLeftReturn_t0; // feature points to check
                                                // circular matching validation
  
  if(currentVOFeatures.size() < 4000) {
      // update feature set with detected features from the image.
      currentVOFeatures.appendFeaturesFromImage(imageLeft_t0, FAST_THRESHOLD);
  }
  if(currentVOFeatures.size() < FEATURES_THRESHOLD * 2) {
      // Just append a bunch of random features
      currentVOFeatures.appendGridOfFeatures(imageLeft_t0);
      // currentVOFeatures.appendFeaturesFromImage(imageLeft_t0, FAST_THRESHOLD / 2);
  }

  // --------------------------------------------------------
  // Feature tracking using KLT tracker, bucketing and circular matching.
  // --------------------------------------------------------

  pointsLeftT0 = currentVOFeatures.points;
  if (currentVOFeatures.points.size() == 0) return; // early exit

  std::vector<bool> matchingStatus = circularMatching(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1, 
                    pointsLeftT0, pointsRightT0, pointsLeftT1, pointsRightT1, pointsLeftReturn_t0);

  // Check if circled back points are in range of original points.
  std::vector<bool> status = findUnmovedPoints(pointsLeftT0, pointsLeftReturn_t0, 1);
  // Only keep points that were matched correctly and are in the image bounds.
  for(unsigned int i = 0; i < status.size(); i++) {
    if(!matchingStatus[i] ||
        (pointsLeftT0[i].x < 0) || (pointsLeftT0[i].y < 0) ||
            (pointsLeftT0[i].y >= imageLeft_t0.rows) || (pointsLeftT0[i].x >= imageLeft_t0.cols) ||
            (pointsLeftT1[i].x < 0) || (pointsLeftT1[i].y < 0) ||
            (pointsLeftT1[i].y >= imageLeft_t1.rows) || (pointsLeftT1[i].x >= imageLeft_t1.cols) ||
            (pointsRightT0[i].x < 0) || (pointsRightT0[i].y < 0) ||
            (pointsRightT0[i].y >= imageRight_t0.rows) || (pointsRightT0[i].x >= imageRight_t0.cols) ||
            (pointsRightT1[i].x < 0) || (pointsRightT1[i].y < 0) ||
            (pointsRightT1[i].y >= imageRight_t1.rows) || (pointsRightT1[i].x >= imageRight_t1.cols)
            // no need to check bounds for pointsLeftReturn_t0 since it's equal to pointsLeftT0 at
            // all valid locations
            ) {
        status[i] = false;
    }
  }

  deleteFeaturesAndPointsWithFailureStatus(
      pointsLeftT0, pointsRightT0, pointsLeftT1, pointsRightT1, pointsLeftReturn_t0,
      currentVOFeatures, status);

  for (unsigned int i = 0; i < currentVOFeatures.ages.size(); ++i) {
    currentVOFeatures.ages[i] += 1;
  }

  // Update current tracked points.
  currentVOFeatures.points = pointsLeftT1;
}
