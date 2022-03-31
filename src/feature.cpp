#include "feature.h"
#include "bucket.h"
#include "vo.h"
#if USE_CUDA
static void download(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void download(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}
#endif


void featureDetectionFast(cv::Mat image, std::vector<cv::Point2f>& points, std::vector<float> & response_strength)  
{   
    //uses FAST as for feature dection, modify parameters as necessary

    std::vector<cv::KeyPoint> keypoints;
    bool nonmaxSuppression = true;
    cv::FAST(image, keypoints, FAST_THRESHOLD, nonmaxSuppression);
    
    // other feature detectors    
    // static cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::StarDetector::create();
    // f2d->detect(image, keypoints);

    cv::KeyPoint::convert(keypoints, points, std::vector<int>());
    response_strength.reserve(points.size());
    for (const auto keypoint : keypoints) response_strength.push_back(keypoint.response); 
}

/* Add more features to feature set using image */
void appendNewFeatures(const cv::Mat& image, FeatureSet& current_features)
{
    /* Fast feature detection */
    std::vector<cv::Point2f>  points_new;
    std::vector<float>  response_strength;

    featureDetectionFast(image, points_new, response_strength);
    current_features.points.insert(current_features.points.end(), points_new.begin(), points_new.end());
    std::vector<int>  ages_new(points_new.size(), 0);
    current_features.ages.insert(current_features.ages.end(), ages_new.begin(), ages_new.end());
    current_features.strengths.insert(current_features.strengths.end(), response_strength.begin(), response_strength.end());

    /* Display feature points after feature detection */
    // displayPoints(image,current_features.points);

    /* Bucketing features */
    const int bucket_size = 1 + std::min(image.rows,image.cols)/BUCKET_DIVISOR; // TODO PARAM

    // filter features in currentVOFeatures so that one per bucket
    bucketingFeatures(image, current_features, bucket_size, FEATURES_PER_BUCKET);
    // debug("[feature]: number of features after bucketing: " + std::to_string(current_features.points.size()));

    /* Display feature points after bucketing */
    // displayPoints(image,current_features.points);
}

void bucketingFeatures(const cv::Mat& image, FeatureSet& current_features, int bucket_size, int features_per_bucket)
{
    // This function buckets features
    // image: only use for getting dimension of the image
    // bucket_size: bucket size in pixel is bucket_size*bucket_size
    // features_per_bucket: number of selected features per bucket
    int image_height = image.rows;
    int image_width = image.cols;
    int buckets_nums_height = image_height/bucket_size + 1;
    int buckets_nums_width = image_width/bucket_size + 1;

    std::vector<Bucket> Buckets;

    // initialize all the buckets
    for (int buckets_idx_height = 0; buckets_idx_height < buckets_nums_height; buckets_idx_height++)
    {
        for (int buckets_idx_width = 0; buckets_idx_width < buckets_nums_width; buckets_idx_width++)
        {
            // Ignore top rows of image.
            if (buckets_idx_height > BUCKET_START_ROW) Buckets.push_back(Bucket(features_per_bucket));
            else Buckets.push_back(Bucket(0));
        }
    }

    /* Put all current features into buckets by their location and scores */
    int buckets_nums_height_idx, buckets_nums_width_idx, buckets_idx;
    for (int i = 0; i < current_features.points.size(); ++i)
    {
        buckets_nums_height_idx = current_features.points[i].y/bucket_size;
        buckets_nums_width_idx = current_features.points[i].x/bucket_size;
        buckets_idx = buckets_nums_height_idx*buckets_nums_width + buckets_nums_width_idx;
        Buckets[buckets_idx].add_feature(current_features.points[i], current_features.ages[i], current_features.strengths[i]);
    }

    /* Take features from buckets and put them back into the feature set */
    current_features.clear();
    for (int buckets_idx_height = 0; buckets_idx_height < buckets_nums_height; buckets_idx_height++)
    {
        for (int buckets_idx_width = 0; buckets_idx_width < buckets_nums_width; buckets_idx_width++)
        {
            buckets_idx = buckets_idx_height*buckets_nums_width + buckets_idx_width;
            FeatureSet bucket_features = Buckets[buckets_idx].features;
            current_features.points.insert(current_features.points.end(), bucket_features.points.begin(), bucket_features.points.end());
            current_features.ages.insert(current_features.ages.end(), bucket_features.ages.begin(), bucket_features.ages.end());
            current_features.strengths.insert(current_features.strengths.end(), bucket_features.strengths.begin(), bucket_features.strengths.end());
        }
    }
}

/* Delete any points that optical flow failed for. */
void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1,
                          std::vector<cv::Point2f>& points2, std::vector<cv::Point2f>& points3,
                          std::vector<cv::Point2f>& points0_return,
                          std::vector<uchar>& status0, std::vector<uchar>& status1,
                          std::vector<uchar>& status2, std::vector<uchar>& status3,
                          FeatureSet & current_features)
    {

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    for (int i = 0; i < current_features.ages.size(); ++i)
    {
        current_features.ages[i] += 1;
    }

    int indexCorrection = 0;
    for( int i=0; i<status3.size(); i++)
    {  
        cv::Point2f pt0 = points0.at(i- indexCorrection);
        cv::Point2f pt1 = points1.at(i- indexCorrection);
        cv::Point2f pt2 = points2.at(i- indexCorrection);
        cv::Point2f pt3 = points3.at(i- indexCorrection);
        cv::Point2f pt0_r = points0_return.at(i- indexCorrection);

        if ((status3.at(i) == 0) || (pt3.x<0) || (pt3.y<0) ||
            (status2.at(i) == 0) || (pt2.x<0) || (pt2.y<0) ||
            (status1.at(i) == 0) || (pt1.x<0) || (pt1.y<0) ||
            (status0.at(i) == 0) || (pt0.x<0) || (pt0.y<0))   
        {
            if((pt0.x<0) || (pt0.y<0) || (pt1.x<0) || (pt1.y<0) 
                || (pt2.x<0) || (pt2.y<0) || (pt3.x<0) || (pt3.y<0))    
            {
                status3.at(i) = 0;
            }
            points0.erase (points0.begin() + (i - indexCorrection));
            points1.erase (points1.begin() + (i - indexCorrection));
            points2.erase (points2.begin() + (i - indexCorrection));
            points3.erase (points3.begin() + (i - indexCorrection));
            points0_return.erase (points0_return.begin() + (i - indexCorrection));

            // also update the feature set 
            current_features.ages.erase (current_features.ages.begin() + (i - indexCorrection));
            current_features.strengths.erase (current_features.strengths.begin() + (i - indexCorrection));

            indexCorrection++;
        }

    }  
}


void featureDetectionGoodFeaturesToTrack(cv::Mat image, std::vector<cv::Point2f>& points)  
{   
    //uses GoodFeaturesToTrack for feature dection, modify parameters as necessary

    int maxCorners = 5000;
    double qualityLevel = 0.01;
    double minDistance = 5.;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    cv::Mat mask;

    cv::goodFeaturesToTrack( image, points, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );
}


//this function automatically gets rid of points for which tracking fails   
void circularMatching(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1, cv::Mat img_r_1,
                      std::vector<cv::Point2f>& points_l_0, std::vector<cv::Point2f>& points_r_0,
                      std::vector<cv::Point2f>& points_l_1, std::vector<cv::Point2f>& points_r_1,
                      std::vector<cv::Point2f>& points_l_0_return,
                      FeatureSet& current_features) { 
      std::vector<float> err;         

    cv::Size winSize=cv::Size(20,20); // Lucas-Kanade optical flow window size                                                                                          
    cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

    std::vector<uchar> status0;
    std::vector<uchar> status1;
    std::vector<uchar> status2;
    std::vector<uchar> status3;

    //clock_t tic = clock();
    // sparse iterative version of the Lucas-Kanade optical flow in pyramids
    calcOpticalFlowPyrLK(img_l_0, img_r_0, points_l_0, points_r_0, status0, err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    calcOpticalFlowPyrLK(img_r_0, img_r_1, points_r_0, points_r_1, status1, err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    calcOpticalFlowPyrLK(img_r_1, img_l_1, points_r_1, points_l_1, status2, err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    calcOpticalFlowPyrLK(img_l_1, img_l_0, points_l_1, points_l_0_return, status3, err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.01);
    //clock_t toc = clock();
    //std::cerr << "calcOpticalFlowPyrLK time: " << float(toc - tic)/CLOCKS_PER_SEC*1000 << "ms" << std::endl;

    deleteUnmatchFeaturesCircle(points_l_0, points_r_0, points_r_1, points_l_1, points_l_0_return,
                        status0, status1, status2, status3, current_features);

}

#if USE_CUDA
void circularMatching_gpu(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1, cv::Mat img_r_1,
                      std::vector<cv::Point2f>& points_l_0, std::vector<cv::Point2f>& points_r_0,
                      std::vector<cv::Point2f>& points_l_1, std::vector<cv::Point2f>& points_r_1,
                      std::vector<cv::Point2f>& points_l_0_return,
                      FeatureSet& current_features) { 
  
    //this function automatically gets rid of points for which tracking fails
                    
    cv::Size winSize=cv::Size(21,21);                                                                                             

    std::vector<uchar> status0;
    std::vector<uchar> status1;
    std::vector<uchar> status2;
    std::vector<uchar> status3;

    clock_t tic_gpu = clock();
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
            winSize, 3, 30);
    cv::cuda::GpuMat img_l_0_gpu(img_l_0);
    cv::cuda::GpuMat img_r_0_gpu(img_r_0);
    cv::cuda::GpuMat img_l_1_gpu(img_l_1);
    cv::cuda::GpuMat img_r_1_gpu(img_r_1);
    cv::cuda::GpuMat status0_gpu(status0);
    cv::cuda::GpuMat status1_gpu(status1);
    cv::cuda::GpuMat status2_gpu(status2);
    cv::cuda::GpuMat status3_gpu(status3);
    cv::cuda::GpuMat points_l_0_gpu(points_l_0);
    cv::cuda::GpuMat points_r_0_gpu(points_r_0);
    cv::cuda::GpuMat points_l_1_gpu(points_l_1);
    cv::cuda::GpuMat points_r_1_gpu(points_r_1);
    cv::cuda::GpuMat points_l_0_ret_gpu(points_l_0_return);

    d_pyrLK_sparse->calc(img_l_0_gpu, img_r_0_gpu, points_l_0_gpu, points_r_0_gpu, status0_gpu);
    d_pyrLK_sparse->calc(img_r_0_gpu, img_r_1_gpu, points_r_0_gpu, points_r_1_gpu, status1_gpu);
    d_pyrLK_sparse->calc(img_r_1_gpu, img_l_1_gpu, points_r_1_gpu, points_l_1_gpu, status2_gpu);
    d_pyrLK_sparse->calc(img_l_1_gpu, img_l_0_gpu, points_l_1_gpu, points_l_0_ret_gpu, status3_gpu);

    download(status0_gpu, status0);
    download(status1_gpu, status1);
    download(status2_gpu, status2);
    download(status3_gpu, status3);
    download(points_l_0_gpu, points_l_0);
    download(points_l_1_gpu, points_l_1);
    download(points_r_0_gpu, points_r_0);
    download(points_r_1_gpu, points_r_1);
    download(points_l_0_ret_gpu, points_l_0_return);

    clock_t toc_gpu = clock();
    std::cerr << "calcOpticalFlowPyrLK(CUDA)  time: " << float(toc_gpu - tic_gpu)/CLOCKS_PER_SEC*1000 << "ms" << std::endl;

    deleteUnmatchFeaturesCircle(points_l_0, points_r_0, points_r_1, points_l_1, points_l_0_return,
                        status0, status1, status2, status3, current_features.ages);
}
#endif


void displayTwoImages(const cv::Mat& image_1, const cv::Mat& image_2)
{
    cv::Size sz1 = image_1.size();
    cv::Size sz2 = image_2.size();
    cv::Mat image_3(sz1.height, sz1.width+sz2.width, CV_8UC1);
    cv::Mat left(image_3, cv::Rect(0, 0, sz1.width, sz1.height));
    image_1.copyTo(left);
    cv::Mat right(image_3, cv::Rect(sz1.width, 0, sz2.width, sz2.height));
    image_2.copyTo(right);
    cv::imshow("im3", image_3);
    cv::waitKey(1);

}

void displayPoints(const cv::Mat& image, const std::vector<cv::Point2f>&  points)
{
    int radius = 2;
    cv::Mat vis;

    cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR, 3);

    for (int i = 0; i < points.size(); i++)
    {
        cv::circle(vis, cv::Point(points[i].x, points[i].y), radius, CV_RGB(0,255,0));
    }

    cv::imshow("vis ", vis );  
    cv::waitKey(1);
}

void displayTracking(const cv::Mat& imageLeft_t1, 
                     const std::vector<cv::Point2f>&  pointsLeft_t0,
                     const std::vector<cv::Point2f>&  pointsLeft_t1)
{
    // -----------------------------------------
    // Display feature racking
    // -----------------------------------------
    int radius = 2;
    cv::Mat vis;

    cv::cvtColor(imageLeft_t1, vis, cv::COLOR_GRAY2BGR, 3);

    for (int i = 0; i < pointsLeft_t0.size(); i++)
    {
      cv::circle(vis, cv::Point(pointsLeft_t0[i].x, pointsLeft_t0[i].y), radius, CV_RGB(0,255,0));
    }

    for (int i = 0; i < pointsLeft_t1.size(); i++)
    {
      cv::circle(vis, cv::Point(pointsLeft_t1[i].x, pointsLeft_t1[i].y), radius, CV_RGB(255,0,0));
    }

    for (int i = 0; i < pointsLeft_t1.size(); i++)
    {
      cv::line(vis, pointsLeft_t0[i], pointsLeft_t1[i], CV_RGB(0,255,0));
    }

    cv::imshow("vis ", vis );  
    cv::waitKey(1);
}