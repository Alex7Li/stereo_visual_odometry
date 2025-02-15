#include "vo.h"

using namespace visual_odometry;

VisualOdometry::VisualOdometry() {
}

void VisualOdometry::initalize_projection_matricies(
    const cv::Mat_<float32> leftCameraProjection, const cv::Mat_<float32> rightCameraProjection) {
    // CORRECT
    // Copy the matrix in case the caller matricies are allocated on the stack,
    // since they may go out of scope
    leftCameraProjection.copyTo(leftCameraProjection_);
    rightCameraProjection.copyTo(rightCameraProjection_);

    left_camera_matrix =
        (cv::Mat_<float32>(3, 3) << leftCameraProjection_.at<float32>(0, 0),
         leftCameraProjection_.at<float32>(0, 1),
         leftCameraProjection_.at<float32>(0, 2),
         leftCameraProjection_.at<float32>(1, 0),
         leftCameraProjection_.at<float32>(1, 1),
         leftCameraProjection_.at<float32>(1, 2),
         leftCameraProjection_.at<float32>(2, 0),
         leftCameraProjection_.at<float32>(2, 1),
         leftCameraProjection_.at<float32>(2, 2));
}

void VisualOdometry::displayPoints(cv::Mat& image, const std::vector<cv::Point2f>&  points)
{
    int radius = 2;
    // Copy image since if we modify the image it will affect the feature detection
    // part of our algorith!
    // cv::Mat out(image.size().height, image.size().width, CV_8UC1);
    // image.copyTo(out);
    for (size_t i = 0; i < points.size(); i++)
    {
        cv::circle(image, cv::Point(points[i].x, points[i].y), radius, 255);
    }
}

std::pair<bool, cv::Mat_<float64>> VisualOdometry::stereo_callback(
    const cv::Mat &imageLeft, const cv::Mat &imageRight) {
    std::pair<bool, cv::Mat> fail_result =
        std::make_pair(false, last_transform);
    // Wait until we have at least two time steps of data
    // to begin predicting the change in pose.
    if (frame_id == 0) {
        imageLeftT0_  = imageLeft;
        imageRightT0_ = imageRight;
        cv::buildOpticalFlowPyramid(imageLeft, lastLeftPyramid, winSize,
                                    maxLevel);
        cv::buildOpticalFlowPyramid(imageRight, lastRightPyramid, winSize,
                                    maxLevel);
        frame_id++;
        return fail_result;
    }
    frame_id++;

    imageLeftT1_  = imageLeft;
    imageRightT1_ = imageRight;

    std::vector<cv::Point2f> pointsLeftT0, pointsRightT0, pointsLeftT1,
        pointsRightT1;

    matchingFeatures(imageLeftT0_, imageRightT0_, imageLeftT1_, imageRightT1_,
                     currentVOFeatures, pointsLeftT0, pointsRightT0,
                     pointsLeftT1, pointsRightT1);

    // Update current tracked points.
    for (unsigned int i = 0; i < currentVOFeatures.ages.size(); ++i) {
        currentVOFeatures.ages[i] += 1;
    }

    imageLeftT1_.copyTo(imageLeftT0_);
    imageRightT1_.copyTo(imageRightT0_);
#if defined(POSE_DEBUG_LOG_FILE)
    displayPoints(imageRightT1_, pointsRightT0);
    displayPoints(imageLeftT1_, pointsLeftT0);
#endif   // POSE_DEBUG_LOG_FILE

    // Need at least 4 points for cameraToWorld to not crash.
    if (pointsLeftT0.size() <= std::max(4, FEATURES_THRESHOLD)) {
        return fail_result;
    }
    // ---------------------
    // Triangulate 3D Points
    // ---------------------
    CFE_ES_WriteToSysLog("Pose Estimator: Start triangulation\n");
    cv::Mat world_points_T0, world_homogenous_points_T0;
    cv::triangulatePoints(leftCameraProjection_, rightCameraProjection_,
                          pointsLeftT0, pointsRightT0,
                          world_homogenous_points_T0);
    cv::convertPointsFromHomogeneous(world_homogenous_points_T0.t(),
                                     world_points_T0);

    // ---------------------
    // Tracking transfomation
    // ---------------------

    CFE_ES_WriteToSysLog("Pose Estimator: Start cameratoworld\n");
    auto result   = cameraToWorld(left_camera_matrix, pointsLeftT1,
                                  world_points_T0, rotation, translation);
    int n_inliers = result.first.size().height;
    bool success  = result.second;

    if (n_inliers < FEATURES_THRESHOLD || !success) {
        if(success){
          CFE_ES_WriteToSysLog("Pose Estimator: Failed, n_inliers %d\n", n_inliers);
        } else {
          CFE_ES_WriteToSysLog("Pose Estimator: RANSAC failed\n");
        }
        return fail_result;
    }

    std::vector<bool> is_ok(pointsLeftT1.size());
    for (int i = 0; i < n_inliers; i++) {
        size_t inlier_index = result.first.at<int>(i);
        is_ok[inlier_index] = true;
    }
    currentVOFeatures.points = pointsLeftT1;
    deleteFeaturesWithFailureStatus(currentVOFeatures, is_ok);

    cv::Mat rotation_rodrigues;
    float64 translation_norm = cv::norm(translation);
    cv::Rodrigues(rotation, rotation_rodrigues);
    float64 angle = cv::norm(rotation_rodrigues, cv::NORM_L2);

    CFE_ES_WriteToSysLog("Pose Estimator: Translation norm of %f\n", translation_norm);
    if (translation_norm > MAX_TRANSLATION_NORM || angle > MAX_ROTATION_NORM) {
        CFE_ES_WriteToSysLog("Pose Estimator: Too big translation norm of %f or rotation norm of %f\n", translation_norm, angle);
        return fail_result;
    }
    cv::Mat transform =
        visual_odometry::getInverseTransform(rotation, translation);
    last_transform = transform;
    return std::make_pair(true, transform);
}

// ---------------------------------------------------------------------------
// Original source for the code in this section
// https://github.com/hjamal3/stereo_visual_odometry/blob/main/src/feature.cpp
// ---------------------------------------------------------------------------

void visual_odometry::deletePointsWithFailureStatus(
    std::vector<cv::Point2f> &point_vector, const std::vector<bool> &is_ok) {
    size_t indexCorrection = 0;
    for (size_t i = 0; i < is_ok.size(); i++) {
        if (!is_ok.at(i)) {
            point_vector.erase(point_vector.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}
void visual_odometry::deleteFeaturesWithFailureStatus(
    FeatureSet &currentFeatures, const std::vector<bool> &is_ok) {
    size_t indexCorrection = 0;
    for (size_t i = 0; i < is_ok.size(); i++) {
        if (!is_ok.at(i)) {
            currentFeatures.ages.erase(currentFeatures.ages.begin() +
                                       (i - indexCorrection));
            currentFeatures.strengths.erase(currentFeatures.strengths.begin() +
                                            (i - indexCorrection));
            currentFeatures.points.erase(currentFeatures.points.begin() +
                                         (i - indexCorrection));
            indexCorrection++;
        }
    }
}
void VisualOdometry::circularMatching(const cv::Mat &imgLeftT1,
                                      const cv::Mat &imgRightT1,
                                      std::vector<cv::Point2f> &pointsLeftT0,
                                      std::vector<cv::Point2f> &pointsRightT0,
                                      std::vector<cv::Point2f> &pointsLeftT1,
                                      std::vector<cv::Point2f> &pointsRightT1,
                                      FeatureSet &current_features) {
    // Check that there are points to match. All features will initially be
    // placed in pointsLeftT0; if this is empty then there are none so we
    // can exit early to avoid crashing.
    if (pointsLeftT0.size() == 0) {
        return;
    }
    std::vector<float32> err;
    cv::TermCriteria termcrit = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.0001);

    std::vector<uchar> status0;
    std::vector<uchar> status1;
    std::vector<uchar> status2;
    std::vector<uchar> status3;
    std::vector<cv::Mat> pyramidl1;
    std::vector<cv::Mat> pyramidr1;
    float64 minEigThreshold = OPTICAL_FLOW_MIN_EIG_THRESHOLD;
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
                         status0, err, winSize, maxLevel, termcrit, 0,
                         minEigThreshold);
    calcOpticalFlowPyrLK(pyramidl1, pyramidr1, pointsLeftT1, pointsRightT1,
                         status1, err, winSize, maxLevel, termcrit, 0,
                         minEigThreshold);
    calcOpticalFlowPyrLK(pyramidr1, lastRightPyramid, pointsRightT1,
                         pointsRightT0, status2, err, winSize, maxLevel,
                         termcrit, 0, minEigThreshold);
    std::vector<cv::Point2f> points_left_T0_circle;
    calcOpticalFlowPyrLK(lastRightPyramid, lastLeftPyramid, pointsRightT0,
                         points_left_T0_circle, status3, err, winSize, maxLevel,
                         termcrit, 0, minEigThreshold);
    // Remove all features that optical flow failed to track.
    std::vector<bool> is_ok =
        findClosePoints(pointsLeftT0, points_left_T0_circle,
                        CIRCULAR_MATCHING_SUCCESS_THRESHOLD);
    int c = 0;
    for (size_t i = 0; i < pointsLeftT1.size(); i++) {
        if(status0[i]){
          c+=1;
        }
    }
    CFE_ES_WriteToSysLog("Pose Estimator: Features returned after first flow matching %d\n", c);
    for (size_t i = 0; i < is_ok.size(); i++) {
        is_ok[i] =
            status0[i] && status1[i] && status2[i] && status3[i] && is_ok[i];
    }
    lastLeftPyramid  = pyramidl1;
    lastRightPyramid = pyramidr1;
    deleteFeaturesWithFailureStatus(current_features, is_ok);
    deletePointsWithFailureStatus(pointsLeftT0, is_ok);
    deletePointsWithFailureStatus(pointsLeftT1, is_ok);
    deletePointsWithFailureStatus(pointsRightT1, is_ok);
    deletePointsWithFailureStatus(pointsRightT0, is_ok);
    deletePointsWithFailureStatus(points_left_T0_circle, is_ok);
    CFE_ES_WriteToSysLog("Pose Estimator: Features returned after all matching %d\n", current_features.size());
}

// --------------------------------
// Original source for the code in this section
// https://github.com/hjamal3/stereo_visual_odometry/blob/main/src/utils.cpp
// --------------------------------
cv::Mat visual_odometry::getInverseTransform(
    const cv::Mat &rotation, const cv::Mat &translation_stereo) {
    cv::Mat rigid_body_transformation;
    // Construct a homogeneous transform from the rotation and translation
    cv::Mat final_homogeneous_row = (cv::Mat_<float64>(1, 4) << 0, 0, 0, 1);

    cv::hconcat(rotation, translation_stereo, rigid_body_transformation);
    cv::vconcat(rigid_body_transformation, final_homogeneous_row,
                rigid_body_transformation);

    rigid_body_transformation = rigid_body_transformation.inv();
    return rigid_body_transformation;
}

// --------------------------------
// Original source for the code in this section
// https://github.com/hjamal3/stereo_visual_odometry/bl_<float32>ob/main/src/visualOdometry.cpp
// --------------------------------

std::vector<bool> visual_odometry::findClosePoints(
    const std::vector<cv::Point2f> &points_1,
    const std::vector<cv::Point2f> &points_2, float32 threshold) {
    std::vector<bool> is_ok;
    float32 offset;
    for (unsigned int i = 0; i < points_1.size(); i++) {
        offset = std::max(std::abs(points_1[i].x - points_2[i].x),
                          std::abs(points_1[i].y - points_2[i].y));
        if (offset > threshold) {
            is_ok.push_back(false);
        } else {
            is_ok.push_back(true);
        }
    }
    return is_ok;
}

std::pair<cv::Mat, bool> visual_odometry::cameraToWorld(
    const cv::Mat_<float32> &left_camera_matrix,
    const std::vector<cv::Point2f> &cameraPoints, const cv::Mat_<float32> &worldPoints,
    cv::Mat &rotation, cv::Mat &translation) {
    // Calculate frame to frame transformation.
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_32F);
    cv::Mat rvec;
    cv::Rodrigues(rotation, rvec);
    // Number of Ransac iterations.
    int iterationsCount = RANSAC_ITERATIONS;
    // maximum allowed distance to consider it an inlier.
    float32 reprojectionError = RANSAC_REPROJECTION_ERROR;
    // RANSAC successful confidence.
    float32 confidence       = 0.98;
    bool useExtrinsicGuess = true;
    int flags              = cv::SOLVEPNP_ITERATIVE;

    cv::Mat inliers;
    cv::Mat old_translation = translation.clone();
    bool success = cv::solvePnPRansac(
                   worldPoints, cameraPoints, left_camera_matrix, distCoeffs, rvec,
                   translation, useExtrinsicGuess, iterationsCount, reprojectionError,
                   confidence, inliers, flags);

    // Save the new rotation and translation only on success.
    if (success) {
        cv::Rodrigues(rvec, rotation);
    } else {
        translation = old_translation;
    }
    return std::make_pair(inliers, success);
}

void VisualOdometry::matchingFeatures(const cv::Mat &imageLeftT0,
                                      const cv::Mat &imageRightT0,
                                      const cv::Mat &imageLeftT1,
                                      const cv::Mat &imageRightT1,
                                      FeatureSet &currentVOFeatures,
                                      std::vector<cv::Point2f> &pointsLeftT0,
                                      std::vector<cv::Point2f> &pointsRightT0,
                                      std::vector<cv::Point2f> &pointsLeftT1,
                                      std::vector<cv::Point2f> &pointsRightT1) {
    // Update feature set with detected features from the first image.
    currentVOFeatures.appendFeaturesFromImage(imageLeftT0, FAST_THRESHOLD);
    CFE_ES_WriteToSysLog("Pose Estimator: Appending Features, found %d\n", currentVOFeatures.size());
    if (currentVOFeatures.size() < PRE_MATCHING_FEATURE_THRESHOLD) {
        // Could not detect enough features, try again with a lower threshold
        currentVOFeatures.appendFeaturesFromImage(imageLeftT0,
                                                  FAST_THRESHOLD / 4);
      CFE_ES_WriteToSysLog("Pose Estimator: Appending Features again, now found %d\n", currentVOFeatures.size());
    }

    // --------------------------------------------------------
    // Feature tracking using KLT tracker, bucketing and circular matching.
    // --------------------------------------------------------

    pointsLeftT0 = currentVOFeatures.points;
    circularMatching(imageLeftT1, imageRightT1, pointsLeftT0, pointsRightT0,
                     pointsLeftT1, pointsRightT1, currentVOFeatures);
    std::vector<bool> is_ok(pointsRightT0.size(), true);
    // Check if circled back points are in range of original points.
    // Only keep points that were matched correctly and are in the image bounds.
    for (unsigned int i = 0; i < is_ok.size(); i++) {
        if ((pointsLeftT0[i].x < 0) || (pointsLeftT0[i].y < 0) ||
            (pointsLeftT0[i].y >= imageLeftT0.rows) ||
            (pointsLeftT0[i].x >= imageLeftT0.cols) ||
            (pointsLeftT1[i].x < 0) || (pointsLeftT1[i].y < 0) ||
            (pointsLeftT1[i].y >= imageLeftT1.rows) ||
            (pointsLeftT1[i].x >= imageLeftT1.cols) ||
            (pointsRightT0[i].x < 0) || (pointsRightT0[i].y < 0) ||
            (pointsRightT0[i].y >= imageRightT0.rows) ||
            (pointsRightT0[i].x >= imageRightT0.cols) ||
            (pointsRightT1[i].x < 0) || (pointsRightT1[i].y < 0) ||
            (pointsRightT1[i].y >= imageRightT1.rows) ||
            (pointsRightT1[i].x >= imageRightT1.cols)) {
            is_ok[i] = false;
        }
    }
    deleteFeaturesWithFailureStatus(currentVOFeatures, is_ok);
    deletePointsWithFailureStatus(pointsLeftT0, is_ok);
    deletePointsWithFailureStatus(pointsLeftT1, is_ok);
    deletePointsWithFailureStatus(pointsRightT0, is_ok);
    deletePointsWithFailureStatus(pointsRightT1, is_ok);
    CFE_ES_WriteToSysLog("Pose Estimator: Features not out of range %d\n", currentVOFeatures.size());
}
