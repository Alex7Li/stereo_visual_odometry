#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2
import rosbag
from cv_bridge import CvBridge
# /stereo/left/image_raw
# /stereo/right/image_raw
def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    print("Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          args.image_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    os.mkdir(args.output_dir)
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        cv2.imwrite(os.path.join(args.output_dir, "frame%06i.png" % count), cv_img)
        print("Wrote image %i" % count)

        count += 1
    if count == 0:
        print("nothing written, make sure your topic is valid?")
        topics = bag.get_type_and_topic_info()[1].keys()
        print(topics)

    bag.close()

    return

if __name__ == '__main__':
    main()