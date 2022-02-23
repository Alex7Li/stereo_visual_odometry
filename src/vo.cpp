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

namespace visual_odometry {
VisualOdometry::VisualOdometry(const cv::Mat leftCameraProjection,
                               const cv::Mat rightCameraProjection) {
  leftCameraProjection_ = leftCameraProjection;
  rightCameraProjection_ = rightCameraProjection;
}

VisualOdometry::~VisualOdometry() {}
void VisualOdometry::stereo_callback_(const cv::Mat &imageLeft,
                                      const cv::Mat &imageRight) {
    // Wait until we have at least two time steps of data
    // to begin predicting the change in pose.
    if (!frame_id) {
      imageLeftT0_ = imageLeft;
      imageRightT0_ = imageRight;
      frame_id++;
      return;
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

    if (currentVOFeatures.size() < 5) {
      // There are not enough features to fully determine
      // equations for pose estimation, so presume nothing and exit.
      frame_id++;
      return;
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
    cameraToWorld(leftCameraProjection_,
        pointsLeftT1, world_points_T0, rotation, translation);

    // ------------------------------------------------
    // Integrating
    // ------------------------------------------------
    cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);
    // Don't perform an update if the output is unusually large, indicates a error elsewhere.
    if (abs(rotation_euler[1]) < 0.1 && abs(rotation_euler[0]) < 0.1 &&
        abs(rotation_euler[2]) < 0.1) {
      integrateOdometryStereo(frame_id, frame_pose, rotation, translation);
    }
    cv::Mat xyz = frame_pose.col(3).clone();
    cv::Mat R = frame_pose(cv::Rect(0, 0, 3, 3));

    // publish
    if (true) {
        static tf::TransformBroadcaster br;

        tf::Transform transform;
        transform.setOrigin(tf::Vector3(xyz.at<double>(0), xyz.at<double>(1),
                                        xyz.at<double>(2)));
        tf::Quaternion q;
        tf::Matrix3x3 R_tf(
            R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2));
        R_tf.getRotation(q);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(),
                                              "odom", "camera"));

        transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
        tf::Quaternion q2(0.5, -0.5, 0.5, -0.5);
        transform.setRotation(q2);
        br.sendTransform(
            tf::StampedTransform(transform, ros::Time::now(), "map",
            "odom"));
    }
    frame_id++;
  }

  // --------------------------------
  // https://github.com/hjamal3/stereo_visual_odometry/blob/main/src/feature.cpp
  // --------------------------------

  void deleteUnmatchFeaturesCircle(
      std::vector<cv::Point2f> & points0, std::vector<cv::Point2f> & points1,
      std::vector<cv::Point2f> & points2, std::vector<cv::Point2f> & points3,
      std::vector<cv::Point2f> & points4, std::vector<int> & ages,
      const std::vector<uchar> & status_all) {
    // getting rid of points for which the KLT tracking failed or those who have
    // gone outside the frame
    int indexCorrection = 0;
    for (int i = 0; i < status_all.size(); i++) {
      cv::Point2f pt0 = points0.at(i - indexCorrection);
      cv::Point2f pt1 = points1.at(i - indexCorrection);
      cv::Point2f pt2 = points2.at(i - indexCorrection);
      cv::Point2f pt3 = points3.at(i - indexCorrection);
      cv::Point2f pt4 = points4.at(i - indexCorrection);
      // TODO: Why are we even considering the x/y coordinates here, should we do it for pt4, and
      // if we should consider it, should we consider the case where they are out of the image along the
      // image size axes?
      if ((status_all.at(i) == 0) || (pt3.x < 0) || (pt3.y < 0) ||
           (pt2.x < 0) || (pt2.y < 0) ||
           (pt1.x < 0) || (pt1.y < 0) ||
           (pt0.x < 0) || (pt0.y < 0)) {
        points0.erase(points0.begin() + (i - indexCorrection));
        points1.erase(points1.begin() + (i - indexCorrection));
        points2.erase(points2.begin() + (i - indexCorrection));
        points3.erase(points3.begin() + (i - indexCorrection));
        points4.erase(points4.begin() + (i - indexCorrection));

        ages.erase(ages.begin() + (i - indexCorrection));
        indexCorrection++;
      }
    }
  }

  void circularMatching(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1,
                        cv::Mat img_r_1, std::vector<cv::Point2f> & points_l_0,
                        std::vector<cv::Point2f> & points_r_0,
                        std::vector<cv::Point2f> & points_l_1,
                        std::vector<cv::Point2f> & points_r_1,
                        std::vector<cv::Point2f> & points_l_0_return,
                        FeatureSet & current_features) {
    std::vector<float> err;

    cv::Size winSize =
        cv::Size(20, 20); // Lucas-Kanade optical flow window size
    cv::TermCriteria termcrit = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    std::vector<uchar> status0;
    std::vector<uchar> status1;
    std::vector<uchar> status2;
    std::vector<uchar> status3;

    // TODO: Do these 4 vectors neccessarily have the same length? Documentation seems suspicious,
    // and if not, there's a big bug... ~ Alex
    // Sparse iterative version of the Lucas-Kanade optical flow in pyramids.
    calcOpticalFlowPyrLK(img_l_0, img_r_0, points_l_0, points_r_0, status0, err,
                         winSize, 3, termcrit, 0, 0.001);
    calcOpticalFlowPyrLK(img_r_0, img_r_1, points_r_0, points_r_1, status1, err,
                         winSize, 3, termcrit, 0, 0.001);
    calcOpticalFlowPyrLK(img_r_1, img_l_1, points_r_1, points_l_1, status2, err,
                         winSize, 3, termcrit, 0, 0.001);
    calcOpticalFlowPyrLK(img_l_1, img_l_0, points_l_1, points_l_0_return,
                         status3, err, winSize, 3, termcrit, 0, 0.001);

    std::vector<uchar> status_all;
    for(int i = 0; i < status3.size(); i++) {
      status_all[i] = status0[i] | status1[i] | status2[i] | status3[i];
    }

    deleteUnmatchFeaturesCircle(points_l_0, points_r_0, points_r_1, points_l_1,
                                points_l_0_return, current_features.ages, status_all);
    for (int i = 0; i < current_features.ages.size(); ++i) {
      current_features.ages[i] += 1;
    }
  }


  // --------------------------------
  // https://github.com/hjamal3/stereo_visual_odometry/blob/main/src/utils.cpp
  // --------------------------------

  void integrateOdometryStereo(cv::Mat &frame_pose, const cv::Mat &rotation,
                               const cv::Mat &translation_stereo) {
    cv::Mat rigid_body_transformation;

    cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);

    cv::hconcat(rotation, translation_stereo, rigid_body_transformation);
    cv::vconcat(rigid_body_transformation, addup, rigid_body_transformation);

    const double scale = sqrt((translation_stereo.at<double>(0)) *
                            (translation_stereo.at<double>(0)) +
                        (translation_stereo.at<double>(1)) *
                            (translation_stereo.at<double>(1)) +
                        (translation_stereo.at<double>(2)) *
                            (translation_stereo.at<double>(2)));

    rigid_body_transformation = rigid_body_transformation.inv();
    if (scale > 0.001 && scale < 10) // WHY DO WE NEED THIS
    {
      frame_pose = frame_pose * rigid_body_transformation;
    } else {
      std::cout << "[WARNING] scale below 0.1, or incorrect translation"
                << std::endl;
    }
  }

  // Calculates rotation matrix to euler angles
  // The result is the same as MATLAB except the order
  // of the euler angles ( x and z are swapped ).
  cv::Vec3f const rotationMatrixToEulerAngles(cv::Mat & R) {
    float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                    R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6;

    float x, y, z;
    if (!singular) {
      x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
      y = atan2(-R.at<double>(2, 0), sy);
      z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
      x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
      y = atan2(-R.at<double>(2, 0), sy);
      z = 0;
    }
    return cv::Vec3f(x, y, z);
  }

  // --------------------------------
  // https://github.com/hjamal3/stereo_visual_odometry/blob/main/src/visualOdometry.cpp
  // --------------------------------

  std::vector<bool> findUnmovedPoints(const std::vector<cv::Point2f> & points_1,
                       const std::vector<cv::Point2f> & points_2,
                       const int threshold) {
    std::vector<bool> status;
    int offset;
    for (int i = 0; i < points_1.size(); i++) {
      offset = std::max(std::abs(points_1[i].x - points_2[i].x),
                        std::abs(points_1[i].y - points_2[i].y));
      if (offset > threshold) {
        status.push_back(false);
      } else {
        status.push_back(true);
      }
    }
    return status
  }

  void removeInvalidPoints(std::vector<cv::Point2f> & points,
                           const std::vector<bool> &status) {
    int index = 0;
    for (int i = 0; i < status.size(); i++) {
      if (status[i] == false) {
        points.erase(points.begin() + index);
      } else {
        index++;
      }
    }
  }
  void cameraToWorld(
      const cv::Mat & cameraProjection,
      const std::vector<cv::Point2f> & cameraPoints, cv::Mat & worldPoints,
      cv::Mat & rotation, cv::Mat & translation) {
    // Calculate frame to frame transformation
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat intrinsic_matrix =
        (cv::Mat_<float>(3, 3) << cameraProjection.at<float>(0, 0),
         cameraProjection.at<float>(0, 1),
         cameraProjection.at<float>(0, 2),
         cameraProjection.at<float>(1, 0),
         cameraProjection.at<float>(1, 1),
         cameraProjection.at<float>(1, 2),
         cameraProjection.at<float>(1, 1),
         cameraProjection.at<float>(1, 2),
         cameraProjection.at<float>(1, 3));

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
  }

  static void matchingFeatures(
      const cv::Mat &imageLeft_t0, const cv::Mat &imageRight_t0,
      const cv::Mat &imageLeft_t1, const cv::Mat &imageRight_t1,
      FeatureSet &currentVOFeatures, std::vector<cv::Point2f> &pointsLeftT0,
      std::vector<cv::Point2f> &pointsRightT0,
      std::vector<cv::Point2f> &pointsLeftT1,
      std::vector<cv::Point2f> &pointsRightT1) {
    
    std::vector<cv::Point2f> pointsLeftReturn_t0; // feature points to check
                                                  // circular matching validation
    // Append new features with old features.
    currentVOFeatures.appendFeaturesFromImage(imageLeft_t0);

    // --------------------------------------------------------
    // Feature tracking using KLT tracker, bucketing and circular matching.
    // --------------------------------------------------------
    int bucket_size =
        std::min(imageLeft_t0.rows, imageLeft_t0.cols) / BUCKETS_PER_AXIS;
    int features_per_bucket = FEATURES_PER_BUCKET;

    // Filter features in currentVOFeatures to leave just one per bucket.
    currentVOFeatures.filterByBucketLocation(imageLeft_t0, bucket_size,
                      features_per_bucket);

    pointsLeftT0 = currentVOFeatures.points;

    circularMatching(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1,
                     pointsLeftT0, pointsRightT0, pointsLeftT1, pointsRightT1,
                     pointsLeftReturn_t0, currentVOFeatures);

    // Check if circled back points are in range of original points.
    std::vector<bool> status = findUnmovedPoints(pointsLeftT0, pointsLeftReturn_t0, status, 0);
    removeInvalidPoints(pointsLeftT0, status);
    removeInvalidPoints(pointsLeftT1, status);
    removeInvalidPoints(pointsRightT0, status);
    removeInvalidPoints(pointsRightT1, status);

    // Update current tracked points.
    currentVOFeatures.points = pointsLeftT1;
  }
} // namespace visual_odometry
