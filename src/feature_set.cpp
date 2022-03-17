#include "vo.h"
namespace visual_odometry {
Bucket::Bucket(int size) { max_size = size; }

Bucket::~Bucket() {}

int Bucket::size() { return features.points.size(); }

/* Score is an integer. Age is from 1-10. Strength goes up to 100. */
int Bucket::compute_score(const int age, const int strength)
{
    return age + (strength - FAST_THRESHOLD)/20;
}

void Bucket::add_feature(const cv::Point2f point, const int age, const int strength) {
  // If max size is none, we never add anything.
  if (!max_size) return;

  const int score = compute_score(age, strength);
  // Don't add sufficently old Features
  if (age < AGE_THRESHOLD) {
    // Insert any feature before bucket is full.
    if (size() < max_size) {
      features.points.push_back(point);
      features.ages.push_back(age);
    } else {
        // Replace the feauture with the lowest score.
        int score_min = compute_score(features.ages[0],features.strengths[0]);
        int score_min_idx = 0;
        for (int i = 1; i < size(); i++)
        {
            const int current_score = compute_score(features.ages[i],features.strengths[i]);
            if (current_score < score_min)
            {
                score_min = current_score;
                score_min_idx = i;
            }
        }
        if (score > score_min)
        {
            features.points[score_min_idx] = point;
            features.ages[score_min_idx] = age;
            features.strengths[score_min_idx] = strength;
        }
    }
  }
}

std::vector<cv::Point2f> featureDetectionFast(const cv::Mat image, std::vector<float> & response_strength) {
  std::vector<cv::Point2f> points;
  std::vector<cv::KeyPoint> keypoints;
  bool nonmaxSuppression = true;
  cv::FAST(image, keypoints, FAST_THRESHOLD, nonmaxSuppression);
  cv::KeyPoint::convert(keypoints, points, std::vector<int>());
  response_strength.reserve(points.size());
  for (const auto keypoint : keypoints) {
    response_strength.push_back(keypoint.response); 
  }
  return points;
}

void FeatureSet::appendFeaturesFromImage(const cv::Mat & image) {
    /* Fast feature detection */
    std::vector<float>  response_strength;
    std::vector<cv::Point2f>  points_new = featureDetectionFast(image, response_strength);

    points.insert(points.end(), points_new.begin(), points_new.end());
    std::vector<int>  ages_new(points_new.size(), 0);
    ages.insert(ages.end(), ages_new.begin(), ages_new.end());
    strengths.insert(strengths.end(), response_strength.begin(), response_strength.end());

    /* Bucketing features */
    const int bucket_size = std::min(image.rows,image.cols)/BUCKETS_PER_AXIS;

    filterByBucketLocation(image, bucket_size);
}

void FeatureSet::filterByBucketLocation(const cv::Mat & image, const int bucket_size) {
    int image_height = image.rows;
    int image_width = image.cols;
    int buckets_nums_height = image_height/bucket_size;
    int buckets_nums_width = image_width/bucket_size;
    int buckets_number = buckets_nums_height * buckets_nums_width;

    std::vector<Bucket> buckets;

    // TODO: Why are these <= rather than <???
    // initialize all the buckets
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
        for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
        {
            // Ignore top rows of image.
            if (buckets_idx_height > BUCKET_START_ROW) buckets.push_back(Bucket(FEATURES_PER_BUCKET));
            else buckets.push_back(Bucket(0));
        }
    }

    /* Put all current features into buckets by their location and scores */
    int buckets_nums_height_idx, buckets_nums_width_idx, buckets_idx;
    for (int i = 0; i < points.size(); ++i)
    {
        buckets_nums_height_idx = points[i].y/bucket_size;
        buckets_nums_width_idx = points[i].x/bucket_size;
        buckets_idx = buckets_nums_height_idx*buckets_nums_width + buckets_nums_width_idx;
        buckets[buckets_idx].add_feature(points[i], ages[i], strengths[i]);
    }

    /* Take features from buckets and put them back into the feature set */
    ages.clear();
    points.clear();
    strengths.clear();
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
        for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
        {
            buckets_idx = buckets_idx_height*buckets_nums_width + buckets_idx_width;
            FeatureSet bucket_features = buckets[buckets_idx].features;
            points.insert(points.end(), bucket_features.points.begin(), bucket_features.points.end());
            ages.insert(ages.end(), bucket_features.ages.begin(), bucket_features.ages.end());
            strengths.insert(strengths.end(), bucket_features.strengths.begin(), bucket_features.strengths.end());
        }
    }
}
} // namespace visual_odometry
