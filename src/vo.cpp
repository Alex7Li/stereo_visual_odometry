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

cv::Mat displayPoints(const cv::Mat& image, const std::vector<cv::Point2f>&  points)
{
    int radius = 2;
    // Copy image since if we modify the image it will affect the feature detection
    // part of our algorith!
    cv::Mat out(image.size().height, image.size().width, CV_8UC1);
    image.copyTo(out);
    for (size_t i = 0; i < points.size(); i++)
    {
        cv::circle(out, cv::Point(points[i].x, points[i].y), radius, 255);
    }
    return out;
}

void displayTracking(const cv::Mat& image_1, 
                     const cv::Mat& image_2,
                     const std::vector<cv::Point2f>&  pointsLeftT0,
                     const std::vector<cv::Point2f>&  pointsRightT1)
{
    // -----------------------------------------
    // Display feature tracking.
    // -----------------------------------------
    int radius = 2;
    cv::Size sz1 = image_1.size();
    cv::Size sz2 = image_2.size();
    cv::Mat image_3(sz1.height, sz1.width+sz2.width, CV_8UC1);
    cv::Mat left(image_3, cv::Rect(0, 0, sz1.width, sz1.height));
    image_1.copyTo(left);
    cv::Mat right(image_3, cv::Rect(sz1.width, 0, sz2.width, sz2.height));
    image_2.copyTo(right);

    cv::Mat vis;
    cv::cvtColor(image_3, vis, cv::COLOR_GRAY2BGR, 3);

    for (size_t i = 0; i < pointsLeftT0.size(); i++)
    {
      cv::circle(vis, cv::Point(pointsLeftT0[i].x, pointsLeftT0[i].y), radius, CV_RGB(0,255,0));
    }

    for (size_t i = 0; i < pointsRightT1.size(); i++)
    {
      cv::circle(vis, cv::Point((int)pointsRightT1[i].x + image_1.size().width, (int)pointsRightT1[i].y), radius, CV_RGB(255,0,0));
    }

    for (size_t i = 0; i < pointsRightT1.size(); i++)
    {
      cv::line(vis, pointsLeftT0[i], {(int)pointsRightT1[i].x  + image_1.size().width, (int)pointsRightT1[i].y}, CV_RGB(0,255,0));
    }

    cv::imshow("vis ", vis );  
    cv::waitKey(0.01);
}

VisualOdometry::VisualOdometry(const cv::Mat leftCameraProjection,
                               const cv::Mat rightCameraProjection) {
  leftCameraProjection_ = leftCameraProjection;
  rightCameraProjection_ = rightCameraProjection;
  left_camera_matrix =
      (cv::Mat_<float>(3, 3) << leftCameraProjection_.at<float>(0, 0), leftCameraProjection_.at<float>(0, 1), leftCameraProjection_.at<float>(0, 2),
        leftCameraProjection_.at<float>(1, 0), leftCameraProjection_.at<float>(1, 1), leftCameraProjection_.at<float>(1, 2),
        leftCameraProjection_.at<float>(2, 0), leftCameraProjection_.at<float>(2, 1), leftCameraProjection_.at<float>(2, 2));

  // Initial angle of the cameras is to face down at 26 degrees
  // double initial_pose[4][4] = {
  //   {0.89879405, 0, 0.43837115, 0},
  //   {0, 1, 0, 0},
  //   {-0.43837115, 0, 0.89879405, 0},
  //   {0, 0, 0, 1},
  // };
  // std::memcpy(frame_pose.data, initial_pose, sizeof(CV_64F) * 16);
}

cv::Vec3f CalculateMean(const cv::Mat_<cv::Vec3f> &points)
{
    if(points.size().height == 0){
      return 0;
    }
    assert(points.size().width == 1);
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    int n_points = points.size().height;
    for(int i = 0; i < n_points; i++){
      double x = double(points(i)[0]);
      double y = double(points(i)[1]);
      double z = double(points(i)[2]);
      mx += x;
      my += y;
      mz += z;
    }
    return cv::Vec3f(mx/n_points, my/n_points, mz/n_points);
}

// source
// https://stackoverflow.com/questions/21206870/opencv-rigid-transformation-between-two-3d-point-clouds
cv::Mat_<double>
FindRigidTransform(const cv::Mat_<cv::Vec3f> &points1, const cv::Mat_<cv::Vec3f> points2)
{
    /* Calculate centroids. */
    cv::Vec3f t1 = CalculateMean(points1);
    cv::Vec3f t2 = CalculateMean(points2);

    cv::Mat_<double> T1 = cv::Mat_<double>::eye(4, 4);
    T1(0, 3) = double(-t1[0]);
    T1(1, 3) = double(-t1[1]);
    T1(2, 3) = double(-t1[2]);

    cv::Mat_<double> T2 = cv::Mat_<double>::eye(4, 4);
    T2(0, 3) = double(t2[0]);
    T2(1, 3) = double(t2[1]);
    T2(2, 3) = double(t2[2]);

    /* Calculate covariance matrix for input points. Also calculate RMS deviation from centroid
     * which is used for scale calculation.
     */
    cv::Mat_<double> C(3, 3, 0.0);
    for (int ptIdx = 0; ptIdx < points1.rows; ptIdx++) {
        cv::Vec3f p1 = points1(ptIdx) - t1;
        cv::Vec3f p2 = points2(ptIdx) - t2;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                C(i, j) += double(p2[i] * p1[j]);
            }
        }
    }

    cv::Mat_<double> u, s, vt;
    cv::SVD::compute(C, s, u, vt);

    cv::Mat_<double> R = u * vt;

    if (cv::determinant(R) < 0) {
        R -= u.col(2) * (vt.row(2) * 2.0);
    }

    cv::Mat_<double> M = cv::Mat_<double>::eye(4, 4);
    R.copyTo(M.colRange(0, 3).rowRange(0, 3));

    cv::Mat_<double> result = T2 * M * T1;
    result /= result(3, 3);
    return result;
}
std::pair<bool, cv::Mat_<double>> RANSACFindRigidTransform(const cv::Mat_<cv::Vec3f> &points1, const cv::Mat_<cv::Vec3f> &points2)
{
  cv::Mat points1Homo;
  cv::convertPointsToHomogeneous(points1, points1Homo);
  int iterations = 200;
  int min_n_points = 3;
  int n_points = points1.size().height;
  if(true){
    std::ofstream outp1;
    std::ofstream outp2;
    outp1.open("run1/p1.csv");
    outp2.open("run1/p2.csv");
    for(int i = 0; i < n_points; i++) {
      outp1 << points1(i) << "\n";
      outp2 << points2(i) << "\n";
    }
  }
  std::vector<int> range(n_points);
  cv::Mat_<double> best;
  int best_inliers = -1;
  // inlier points should be projected within this many meters
  float threshold = .02;
  std::iota(range.begin(), range.end(), 0);
  auto gen = std::mt19937{std::random_device{}()};
  for(int i = 0; i < iterations; i++) {
    std::shuffle(range.begin(), range.end(), gen);
    cv::Mat_<cv::Vec3f> points1subset(min_n_points, 1, cv::Vec3f(0,0,0));
    cv::Mat_<cv::Vec3f> points2subset(min_n_points, 1, cv::Vec3f(0,0,0));
    for(int j = 0; j < min_n_points; j++) {
      points1subset(j) = points1(range[j]);
      points2subset(j) = points2(range[j]);
    }
    cv::Mat_<float> rigidT = FindRigidTransform(points1subset, points2subset);
    cv::Mat_<float> rigidT_float = cv::Mat::eye(4, 4, CV_32F);
    rigidT.convertTo(rigidT_float, CV_32F);
    std::vector<int> inliers;
    for(int j = 0; j < n_points; j++) {
        cv::Mat_<float> t1_3d = rigidT_float * cv::Mat_<float>(points1Homo.at<cv::Vec4f>(j));
        if(t1_3d(3) == 0) {
          continue; // Avoid 0 division
        }
        float dx = (t1_3d(0)/t1_3d(3) - points2(j)[0]);
        float dy = (t1_3d(1)/t1_3d(3) - points2(j)[1]);
        float dz = (t1_3d(2)/t1_3d(3) - points2(j)[2]);
        float square_dist = dx * dx + dy * dy + dz * dz;
        if(square_dist < threshold * threshold){
          inliers.push_back(j);
        }
    }
    int n_inliers = inliers.size();
    if(n_inliers > best_inliers) {
      best_inliers = n_inliers;
      best = rigidT;
    }
  }
  // dbg(best_inliers);
  if(best_inliers < FEATURES_THRESHOLD){
    return std::make_pair(false, best);
  }
  return std::make_pair(true, best);
}

cv::Mat getWorldPoints(const cv::Mat &leftCameraProjection, const cv::Mat & rightCameraProjection,
              std::vector<cv::Point2f> pointsLeft, std::vector<cv::Point2f> pointsRight) {
    cv::Mat world_points, world_homogenous_points;
    cv::triangulatePoints(leftCameraProjection, rightCameraProjection,
                          pointsLeft, pointsRight, world_homogenous_points);
    cv::convertPointsFromHomogeneous(world_homogenous_points.t(), world_points);
    return world_points;
}


VisualOdometry::~VisualOdometry() {}
std::pair<bool, cv::Mat_<double>> VisualOdometry::stereo_callback(
      const cv::Mat &imageLeft, const cv::Mat &imageRight) {
    std::pair<bool, cv::Mat> fail_result = std::make_pair(false,
        cv::Mat::eye(4, 4, CV_64F));
    // Wait until we have at least two time steps of data
    // to begin predicting the change in pose.
    if (!frame_id) {
      imageLeftT0_ = imageLeft;
      imageRightT0_ = imageRight;
      cv::buildOpticalFlowPyramid(imageLeft, lastLeftPyramid, winSize, maxLevel);
      cv::buildOpticalFlowPyramid(imageRight, lastRightPyramid, winSize, maxLevel);
      frame_id++;
      return fail_result;
    }
    frame_id++;

    imageLeftT1_ = imageLeft;
    imageRightT1_ = imageRight;

    std::vector<cv::Point2f> pointsLeftT0, pointsRightT0, pointsLeftT1,
        pointsRightT1;

    matchingFeatures(imageLeftT0_, imageRightT0_, imageLeftT1_, imageRightT1_,
                     currentVOFeatures, pointsLeftT0, pointsRightT0,
                     pointsLeftT1, pointsRightT1);

    cv::Mat left_scene = displayPoints(imageLeftT0_, pointsLeftT0);
    cv::Mat right_scene = displayPoints(imageRightT0_, pointsRightT0);
    // Update current tracked points.
    for (unsigned int i = 0; i < currentVOFeatures.ages.size(); ++i) {
      currentVOFeatures.ages[i] += 1;
    }
    currentVOFeatures.points = pointsLeftT1;

    imageLeftT0_ = imageLeftT1_;
    imageRightT0_ = imageRightT1_;


    displayTracking(left_scene, right_scene, pointsLeftT0, pointsRightT0);
    // Won't be able to find a good rigid transform
    if (pointsLeftT0.size() <= FEATURES_THRESHOLD) {
      frame_id++;
      dbgstr("Not enough points");
      return fail_result;
    }
    // ---------------------
    // Triangulate 3D Points
    // ---------------------
    dbg(pointsLeftT0.size());
    cv::Mat_<cv::Vec3f> world_points_T0 = getWorldPoints(leftCameraProjection_, rightCameraProjection_, pointsLeftT0, pointsRightT0);
    cv::Mat_<cv::Vec3f> world_points_T1 = getWorldPoints(leftCameraProjection_, rightCameraProjection_, pointsLeftT1, pointsRightT1);
    // Remove ridiculous pairs; robot won't move more even close to .1m/sec
    std::vector<int> okLocs;
    for(size_t i = 0; i < pointsLeftT0.size(); i++) {
      if(cv::norm(world_points_T0(i)-world_points_T1(i)) < .1){
        okLocs.push_back(i);
      }
    }
    cv::Mat_<cv::Vec3f> world_points_T0_filter(cv::Size(1, okLocs.size()), cv::Vec3f(0,0,0));
    cv::Mat_<cv::Vec3f> world_points_T1_filter(cv::Size(1, okLocs.size()), cv::Vec3f(0,0,0));
    for(size_t i = 0; i < okLocs.size(); i++) {
      world_points_T0_filter(i) = world_points_T0(okLocs[i]);
      world_points_T1_filter(i) = world_points_T0(okLocs[i]);
    }

    // Find mapping from points from time 1 to time 0
    // std::pair<bool, cv::Mat> result = FindRigidTransformRemovingNoise(world_points_T1, world_points_T0);
    std::pair<bool, cv::Mat_<double>> result = RANSACFindRigidTransform(world_points_T1, world_points_T0);
    cv::Mat_<double> rigidT = result.second;
    double magnitude = 0;
    for(int j = 0; j < 3; j++) {
      magnitude += rigidT(j, 3) * rigidT(j, 3);
    }
    cv::Mat rodrigues;
    cv::Rodrigues(rigidT.colRange(0,3).rowRange(0, 3), rodrigues);
    double angle = cv::norm(rodrigues, cv::NORM_L2);
    if(magnitude > .1 || angle > .5){
      result.first = false;
    }
    return result;
  }

  // --------------------------------
  // https://github.com/hjamal3/stereo_visual_odometry/blob/main/src/feature.cpp
  // --------------------------------

void visual_odometry::deletePointsWithFailureStatus(
  std::vector<cv::Point2f> &point_vector, const std::vector<bool> &isok){
  size_t indexCorrection = 0;
  for (size_t i = 0; i < isok.size(); i++) {
      if (!isok.at(i)) {
        point_vector.erase(point_vector.begin() + (i - indexCorrection));
        indexCorrection++;
      }
    }
}
void visual_odometry::deleteFeaturesWithFailureStatus(
    FeatureSet &currentFeatures, const std::vector<bool> &isok) {
  size_t indexCorrection = 0;
  for (size_t i = 0; i < isok.size(); i++) {
      if (!isok.at(i)) {
        currentFeatures.ages.erase(currentFeatures.ages.begin() + (i - indexCorrection));
        currentFeatures.strengths.erase(currentFeatures.strengths.begin() + (i - indexCorrection));
        currentFeatures.points.erase(currentFeatures.points.begin() + (i - indexCorrection));
        indexCorrection++;
      }
    }
  }

void VisualOdometry::circularMatching(
    const cv::Mat &imgLeftT1,
    const cv::Mat &imgRightT1,
    std::vector<cv::Point2f> &pointsLeftT0,
    std::vector<cv::Point2f> &pointsRightT0,
    std::vector<cv::Point2f> &pointsLeftT1,
    std::vector<cv::Point2f> &pointsRightT1,
    FeatureSet &current_features) {
    if (pointsLeftT0.size() == 0) {
        return; // Avoid edge cases by exiting early.
    }
    std::vector<float> err;
    cv::TermCriteria termcrit = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.0001);

    std::vector<uchar> status0;
    std::vector<uchar> status1;
    std::vector<uchar> status2;
    std::vector<uchar> status3;
    std::vector<cv::Mat> pyramidl1;
    std::vector<cv::Mat> pyramidr1;
    double minEigThreshold = 0.0001;
    // Perform the circular matching in the order
    // leftT0 -> leftT1 -> rightT1 -> rightT0 -> leftT0.
    // There is a reason to go in this order rather than
    // leftT0 -> rightT0 -> rightT1 -> leftT1 -> leftT0:
    // we want the leftt1 points to be as accurate as possible, since
    // they are used to estimate the position of features in the
    // next frame.
    cv::buildOpticalFlowPyramid(imgLeftT1, pyramidl1, winSize, maxLevel);
    cv::buildOpticalFlowPyramid(imgRightT1, pyramidr1, winSize, maxLevel);

    calcOpticalFlowPyrLK(lastLeftPyramid, pyramidl1, pointsLeftT0, pointsLeftT1,
                         status0, err, winSize, maxLevel, termcrit, 0, minEigThreshold);
    calcOpticalFlowPyrLK(pyramidl1, pyramidr1, pointsLeftT1, pointsRightT1,
                         status1, err, winSize, maxLevel, termcrit, 0, minEigThreshold);
    calcOpticalFlowPyrLK(pyramidr1, lastRightPyramid, pointsRightT1, pointsRightT0,
                         status2, err, winSize, maxLevel, termcrit, 0, minEigThreshold);
    std::vector<cv::Point2f> points_left_T0_circle;
    calcOpticalFlowPyrLK(lastRightPyramid, lastLeftPyramid, pointsRightT0, points_left_T0_circle,
                         status3, err, winSize, maxLevel, termcrit, 0, minEigThreshold);
    // Remove all features that optical flow failed to track.
    std::vector<bool> is_ok = findClosePoints(pointsLeftT0, points_left_T0_circle, .15);
    for (size_t i = 0; i < is_ok.size(); i++) {
        is_ok[i] = status0[i] || status1[i] || status2[i] || status3[i] || is_ok[i];
    }
    lastLeftPyramid = pyramidl1;
    lastRightPyramid = pyramidr1;
    deleteFeaturesWithFailureStatus(current_features, is_ok);
    deletePointsWithFailureStatus(pointsLeftT0, is_ok);
    deletePointsWithFailureStatus(pointsLeftT1, is_ok);
    deletePointsWithFailureStatus(pointsRightT1, is_ok);
    deletePointsWithFailureStatus(pointsRightT0, is_ok);
    deletePointsWithFailureStatus(points_left_T0_circle, is_ok);
}


// --------------------------------
// https://github.com/hjamal3/stereo_visual_odometry/blob/main/src/utils.cpp
// --------------------------------
cv::Mat visual_odometry::getInverseTransform(const cv::Mat &rotation,
                              const cv::Mat &translation_stereo) {
  cv::Mat rigid_body_transformation;

  cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);

  cv::hconcat(rotation, translation_stereo, rigid_body_transformation);
  cv::vconcat(rigid_body_transformation, addup, rigid_body_transformation);

  rigid_body_transformation = rigid_body_transformation.inv();
  // frame_pose = frame_pose * rigid_body_transformation;
  return rigid_body_transformation;
}

// --------------------------------
// https://github.com/hjamal3/stereo_visual_odometry/blob/main/src/visualOdometry.cpp
// --------------------------------

std::vector<bool> visual_odometry::findClosePoints(const std::vector<cv::Point2f> & points_1,
                      const std::vector<cv::Point2f> & points_2,
                      float threshold) {
  std::vector<bool> isok;
  float offset;
  for (unsigned int i = 0; i < points_1.size(); i++) {
    offset = std::max(std::abs(points_1[i].x - points_2[i].x),
                      std::abs(points_1[i].y - points_2[i].y));
    if (offset > threshold) {
      isok.push_back(false);
    } else {
      isok.push_back(true);
    }
  }
  return isok;
}

std::pair<cv::Mat, bool> visual_odometry::cameraToWorld(
    const cv::Mat & left_camera_matrix,
    const std::vector<cv::Point2f> & cameraPoints, const cv::Mat & worldPoints,
    cv::Mat & rotation, cv::Mat & translation) {
  // Calculate frame to frame transformation
  cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
  cv::Mat rvec;
  cv::Rodrigues(rotation, rvec);
  int iterationsCount = 100; // number of Ransac iterations.
  float reprojectionError = .1; // maximum allowed distance to consider it an inlier.
  float confidence = 0.9999; // RANSAC successful confidence.
  bool useExtrinsicGuess = true;
  int flags = cv::SOLVEPNP_ITERATIVE;

  cv::Mat inliers;
  bool success = cv::solvePnPRansac(worldPoints, cameraPoints, left_camera_matrix, distCoeffs,
                      rvec, translation, useExtrinsicGuess, iterationsCount,
                      reprojectionError, confidence, inliers, flags);
  cv::Rodrigues(rvec, rotation);
  return std::make_pair(inliers, success);
}

void VisualOdometry::matchingFeatures(
    const cv::Mat &imageLeftT0, const cv::Mat &imageRightT0,
    const cv::Mat &imageLeftT1, const cv::Mat &imageRightT1,
    FeatureSet &VOFeatures, std::vector<cv::Point2f> &pointsLeftT0,
    std::vector<cv::Point2f> &pointsRightT0,
    std::vector<cv::Point2f> &pointsLeftT1,
    std::vector<cv::Point2f> &pointsRightT1) {
  
  // Update feature set with detected features from the first image.
  VOFeatures.appendFeaturesFromImage(imageLeftT0, FAST_THRESHOLD);
  if (VOFeatures.size() < PRE_MATCHING_FEATURE_THRESHOLD) {
    dbg("Adding more features")
    // Lower criteria, we just want more features.
    VOFeatures.appendFeaturesFromImage(imageLeftT0, FAST_THRESHOLD / 2);
  }

  // --------------------------------------------------------
  // Feature tracking using KLT tracker, bucketing and circular matching.
  // --------------------------------------------------------

  pointsLeftT0 = VOFeatures.points;
  circularMatching(imageLeftT1, imageRightT1, 
              pointsLeftT0, pointsRightT0, pointsLeftT1, pointsRightT1,
                    VOFeatures);
  std::vector<bool> is_ok(pointsRightT0.size(), true);
  // Check if circled back points are in range of original points.
  // Only keep points that were matched correctly and are in the image bounds.
  for(unsigned int i = 0; i < is_ok.size(); i++) {
    // check boundary conditions
    if((pointsLeftT0[i].x < 0) || (pointsLeftT0[i].y < 0) ||
            (pointsLeftT0[i].y >= imageLeftT0.rows) || (pointsLeftT0[i].x >= imageLeftT0.cols) ||
            (pointsLeftT1[i].x < 0) || (pointsLeftT1[i].y < 0) ||
            (pointsLeftT1[i].y >= imageLeftT1.rows) || (pointsLeftT1[i].x >= imageLeftT1.cols) ||
            (pointsRightT0[i].x < 0) || (pointsRightT0[i].y < 0) ||
            (pointsRightT0[i].y >= imageRightT0.rows) || (pointsRightT0[i].x >= imageRightT0.cols) ||
            (pointsRightT1[i].x < 0) || (pointsRightT1[i].y < 0) ||
            (pointsRightT1[i].y >= imageRightT1.rows) || (pointsRightT1[i].x >= imageRightT1.cols)
            // Since the images are rectified, all the matched points ought to be
            // horizontal. Though it's not really horizontal when rolling over
            // rocks and stuff, so the threshold is really generous.
            || (abs(pointsLeftT0[i].y - pointsRightT0[i].y) > 100) ||
            (abs(pointsLeftT0[i].y - pointsRightT1[i].y) > 100)
            ) {
        is_ok[i] = false;
    }
  }
  deleteFeaturesWithFailureStatus(VOFeatures, is_ok);
  deletePointsWithFailureStatus(pointsLeftT0, is_ok);
  deletePointsWithFailureStatus(pointsLeftT1, is_ok);
  deletePointsWithFailureStatus(pointsRightT0, is_ok);
  deletePointsWithFailureStatus(pointsRightT1, is_ok);
}