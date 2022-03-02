#include "stereo_visual_odometry/vo.h"

Bucket::Bucket(int size) { max_size = size; }

Bucket::~Bucket() {}

int Bucket::size() { return features.points.size(); }

void Bucket::add_feature(const cv::Point2f point, const int age) {
  // Don't add sufficently old Features
  if (age < AGE_THRESHOLD) {
    // Insert any feature before bucket is full.
    if (size() < max_size) {
      features.points.push_back(point);
      features.ages.push_back(age);
    } else {
      // TODO: Isn't this backwards? Don't we want to remove the
      // oldest feature? ~ Alex
      
      // Insert feature with old age and remove youngest one.
      int age_min = features.ages[0];
      int age_min_idx = 0;

      for (int i = 0; i < size(); i++) {
        if (age < age_min) {
          age_min = age;
          age_min_idx = i;
        }
      }
      features.points[age_min_idx] = point;
      features.ages[age_min_idx] = age;
    }
  }
}

std::vector<cv::Point2f> featureDetectionFast(const cv::Mat image) {
  std::vector<cv::Point2f> points;
  std::vector<cv::KeyPoint> keypoints;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  cv::FAST(image, keypoints, fast_threshold, nonmaxSuppression);
  cv::KeyPoint::convert(keypoints, points, std::vector<int>());
  return points;
}

void FeatureSet::appendFeaturesFromImage(const cv::Mat & image) {
  std::vector<cv::Point2f> points_new = featureDetectionFast(image);
  points.insert(points.end(), points_new.begin(), points_new.end());
  std::vector<int> ages_new(points_new.size(), 0);
  ages.insert(ages.end(), ages_new.begin(), ages_new.end());
}

void FeatureSet::filterByBucketLocation(const cv::Mat & image,
                        const int bucket_size, const int features_per_bucket) {
  int image_height = image.rows;
  int image_width = image.cols;
  int buckets_nums_height = image_height / bucket_size;
  int buckets_nums_width = image_width / bucket_size;
  int buckets_number = buckets_nums_height * buckets_nums_width;

  std::vector<Bucket> buckets(bucket_nums_height * bucket_nums_width,
                  Bucket(features_per_bucket));

  // bucket all current features into buckets by their location
  int buckets_nums_height_idx, buckets_nums_width_idx, buckets_idx;
  for (int i = 0; i < points.size(); ++i) {
    buckets_nums_height_idx = points[i].y / bucket_size;
    buckets_nums_width_idx = points[i].x / bucket_size;
    buckets_idx =
        buckets_nums_height_idx * buckets_nums_width + buckets_nums_width_idx;
    buckets[buckets_idx].add_feature(points[i], ages[i]);
  }

  points.clear();
  ages.clear();


  for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height;
        buckets_idx_height++) {
    for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width;
          buckets_idx_width++) {
      buckets_idx =
          buckets_idx_height * buckets_nums_width + buckets_idx_width;
      bucket = buckets[buckets_idx];
      points.insert(points.end(), features.points.begin(), features.points.end());
      ages.insert(ages.end(), features.ages.begin(), features.ages.end());
      }
  }
}
