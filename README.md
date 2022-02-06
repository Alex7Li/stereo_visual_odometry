# stereo_visual_odometry
Stereo Visual Odometry in ROS  
roslaunch stereo_visual_odometry powerranger.launch 

# Structure

## Entry point: `stereo_vo.cpp`

We first initialize the projection matrix parameters by using the
parameters in the calib_yaml file.
Then we setup a stereo VO object with these projection matricies, and set up a callback
function to periodically update the stereo VO
object with cvImages of two images obtained from the camera.

### StereoVO::run()

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
