# stereo_visual_odometry
Stereo Visual Odometry in ROS  
roslaunch stereo_visual_odometry powerranger.launch 

# Structure

# Entry point: `stereo_vo.cpp`

We first initialize the projection matrix parameters by using the
parameters in the calib_yaml file.
Then we setup a stereo VO object with these projection matricies, and set up a callback
function to periodically update the stereo VO
object with cvImages of two images obtained from the camera.

## StereoVO::run()

first we call `matchingFeatures` to track the
most relevant features with FAST detector. We
use `bucketingFeatures` to ensure a good spread
of features on the image, and use `circularMatching` to ensure that all
the points are on all 4 images [l/r, past/present]

This gives us a set of 2d points in camera coordinates, so a triangulation is used between the
two cameras to map the coordinates into 3d.

However, this is surely a noisy transform. We
use RANSAC in `trackingFrame2Frame` to find the least square transformation from camera into real world coordinates.

If the rotation is small (for some reason it's detected with euler angles instead of the actual
angle change, maybe just bad design?), then we
call `integrateOdometryStereo`, which updates the
current frame pose according to the given transform.

> 📝 `integrateOdometryStereo` doesn't do this if the `scale` variable is very small or big. Checking if scale is small makes very little sense though, because (despite it's name), scale measures the magnitude of the translation vector.

Finally, we send our transform out to update the camera and odometry systems.

## Matching features

This method takes in 4 images and generates a set of features that are in all 4 in the given 4 array pointers.

### Feature Detection

We use the FAST feature detector to detect keypoints and return their xy coordinates in the image.

### Bucketing Features

Make a grid of buckets and add at most a constant number of features per bucket, preferring to remove features detected
more frames in the past.

### Circular Matching

Use the Lucas-Kanade flow algorithm to find out, for 4 image pairs, where the feature
points move when going from one image to another, as well as a status vector for features whose movement could not be detected.

#### Delete Unmatched Features Circle

Updates the ages of the points (this function is not really a place to do it in good style IMO), then do something that
looks like a bug but seems to intend to erase all the points that aren't in each image.

### CheckValidMatch, removeInvalidPoints

The circular matching of the 4 images is an endomorphism of feature points in the t0 image to features that we expect to be the identity. We restrict the domain to enforce this intuition.

## TrackingFrame2Frame

use RANSAC in to find the least square transformation from camera into real world coordinates, storing it as a rotation matrix.

## RotationMatrixToEulerAngles

Figure out the angle through which the unit x, y, and z vectors rotate under the rotation from RANSAC.

##  Integrate Odometry Stereo

Given the rotation and translation matrix,
that have been predicted for this frame,
we construct a 4d matrix mapping a homogenous 3d point into another homogenous 3d point. Then, we use this transform to map
the current pose to the next frame pose.

We also compute how big the translation is,
and report an error with the frame if it's too big. And apparently too small??

Also idk what it has to do with integration




# Morphin
Original in ~/ssd/code/autonomy_kinematis/src/position_estimator/stereo_visual_odometry
Mine in Desktop/alex_ws

```bash
catkin_make
rviz
rosbag play my.bag
rosbag info my.bag

```
