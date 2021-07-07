#!/usr/bin/python
# -*- coding: utf-8 -*-
# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2019 Intel Corporation. All Rights Reserved.
# Python 2/3 compatibility
from __future__ import print_function

"""
This example shows how to use T265 intrinsics and extrinsics in OpenCV to
asynchronously compute depth maps from T265 fisheye images on the host.

T265 is not a depth camera and the quality of passive-only depth options will
always be limited compared to (e.g.) the D4XX series cameras. However, T265 does
have two global shutter cameras in a stereo configuration, and in this example
we show how to set up OpenCV to undistort the images and compute stereo depth
from them.

Getting started with python3, OpenCV and T265 on Ubuntu 16.04:

First, set up the virtual enviroment:

$ apt-get install python3-venv  # install python3 built in venv support
$ python3 -m venv py3librs      # create a virtual environment in pylibrs
$ source py3librs/bin/activate  # activate the venv, do this from every terminal
$ pip install opencv-python     # install opencv 4.1 in the venv
$ pip install pyrealsense2      # install librealsense python bindings

Then, for every new terminal:

$ source py3librs/bin/activate  # Activate the virtual environment
$ python3 t265_stereo.py        # Run the example
"""

# First import the library
import pyrealsense2 as rs
import rospy
import message_filters
import yaml
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
# Import OpenCV and numpy
import cv2
import numpy as np
from math import tan, pi

"""
In this section, we will set up the functions that will translate the camera
intrinsics and extrinsics from librealsense into parameters that can be used
with OpenCV.

The T265 uses very wide angle lenses, so the distortion is modeled using a four
parameter distortion model known as Kanalla-Brandt. OpenCV supports this
distortion model in their "fisheye" module, more details can be found here:

https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
"""
# R = np.array([[  0.999973714,  -0.0025165,        -0.006799683],
#                 [0.0025133663,  0.999973714,      -0.0004657713],
#                 [0.0068008313,  0.000448668026,    0.999973714]])
# T = np.array([[  -0.06417236, 0.00037436, -0.00021799]])

class Undistort_T265:
    def __init__(self):
        """ 
        Initialize undistortion test, define subscribers
        """
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Undistort Tracking Camera Started")
        self.stereo_subscriber()
        _ = rospy.wait_for_message("camera/fisheye1/image_raw", Image)
        _ = rospy.wait_for_message("camera/fisheye2/image_raw", Image)
        self.left_img_pub = rospy.Publisher(
            "camera/fisheye1/undistort_img", Image, queue_size=1)
        self.right_img_pub = rospy.Publisher(
            "camera/fisheye2/undistort_img", Image, queue_size=1)
        self.start()
        # rospy.spin()

    def stereo_subscriber(self):
        """
        Define the Subscriber with time synchronization among the image topics
        from the stereo camera
        """
        left_img_sub = message_filters.Subscriber(
            "camera/fisheye1/image_raw", Image)
        # print(left_img_sub)
        left_cam_info_sub = message_filters.Subscriber(
            "camera/fisheye1/camera_info", CameraInfo)
        right_img_sub = message_filters.Subscriber(
            "camera/fisheye2/image_raw", Image)
        right_cam_info_sub = message_filters.Subscriber(
            "camera/fisheye2/camera_info", CameraInfo)
        #ts = message_filters.ApproximateTimeSynchronizer([left_img_sub,left_cam_info_sub,right_img_sub,right_cam_info_sub],10, 0.1, allow_headerless=True)
        ts = message_filters.ApproximateTimeSynchronizer(
            [left_img_sub, right_img_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.image_callback)

    def image_callback(self, left_img, right_img):
        """
        Subscriber callback for the stereo camera, with synchronized images

        **Maybe dont call start from image callback??????********
        """
        self.left_img = left_img
        # self.left_cam_info = left_cam_info
        self.right_img = right_img
        # self.right_cam_info = right_cam_info
        # self.start()
    """
    Returns R, T transform from src to dst
    """
    def get_extrinsics(self, src, dst):
        # extrinsics = src.get_extrinsics_to(dst)
        # R = np.reshape(extrinsics.rotation, [3, 3]).T
        # T = np.array(extrinsics.translation)

        R = np.array([[  0.999973714,  -0.00251647457,     -0.006799683],
                [0.0025133663,          0.999973714,       -0.0004657713],
                [0.0068008313,          0.000448668026,     0.999973714]])

        T = np.array([[  -0.06417236, 0.00037436, -0.00021799]])

        print("-- R: ", R)
        print("-- T: ", T)
        print("-- T[0]: ", T[0])
        return (R, T)

    """
    Returns a camera matrix K from librealsense intrinsics
    """
    def camera_matrix(intrinsics):
        return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                         [0, intrinsics.fy, intrinsics.ppy],
                         [0,             0,              1]])

    """
    Returns the fisheye distortion from librealsense intrinsics
    """
    def fisheye_distortion(intrinsics):
        return np.array(intrinsics.coeffs[:4])

    # Set up a mutex to share data between threads
    from threading import Lock
    # frame_mutex = Lock()
    frame_data = {"left": None,
                  "right": None,
                  "timestamp_ms": None
                  }

# -----------------------------------------------------------------------------------------------------

    def start(self):
        """
        Loop through subscribed img msgs, undistort and
        republish to ud_img_msg
        """

        # self.left_img = False
        while not rospy.is_shutdown():
            if (self.left_img) and (self.left_img.header.seq != -1):
                # print(self.left_img.header)
                self.left_img.header.seq = -1
                self.bridge = CvBridge()
                # (self.left_img, desired_encoding='passthrough')
                original_left_img = self.bridge.imgmsg_to_cv2(
                    self.left_img, "rgb8")
                original_right_img = self.bridge.imgmsg_to_cv2(
                    self.right_img, "rgb8")

                # original_left_img = original_left_img.astype('uint8')
                # original_right_img = original_right_img.astype('uint8')


                resized_left_img = cv2.resize(original_left_img, dsize=(
                    300, 300), interpolation=cv2.INTER_CUBIC)
                resized_right_img = cv2.resize(original_right_img, dsize=(
                    300, 300), interpolation=cv2.INTER_CUBIC)

        # try:
            # Set up an OpenCV window to visualize the results
            WINDOW_TITLE = 'Realsense'
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

            # Configure the OpenCV stereo algorithm. See
            # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html for a
            # description of the parameters
            window_size = 5
            min_disp = 0
            # must be divisible by 16
            num_disp = 112 - min_disp
            max_disp = min_disp + num_disp
            stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                           numDisparities=num_disp,
                                           blockSize=16,
                                           P1=8*3*window_size**2,
                                           P2=32*3*window_size**2,
                                           disp12MaxDiff=1,
                                           uniquenessRatio=10,
                                           speckleWindowSize=100,
                                           speckleRange=32)

            # Retreive the stream and intrinsic properties for both cameras
            # profiles = pipe.get_active_profile()
            # streams = {"left"  : profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
            #            "right" : profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}

            # streams: ROS version (images)

            # intrinsics = {"left"  : streams["left"].get_intrinsics(),
            #               "right" : streams["right"].get_intrinsics()}

            # Print information about both cameras
            # print("Left camera:",  intrinsics["left"])
            # print("Right camera:", intrinsics["right"])

            K_left = rospy.get_param("/camera_matrix/data")
            K_left = np.reshape(K_left, [3,3])
            print("K_left reshape: ",K_left)
            D_left = rospy.get_param("/distortion_coefficients/data")
            D_left = np.reshape(D_left, [1,4])
            print("D",D_left)
            K_right = rospy.get_param("/camera_matrix/data")
            K_right = np.reshape(K_left, [3,3])

            D_right = rospy.get_param("/distortion_coefficients/data")
            D_right = np.reshape(D_left, [1,4])
           
            # Translate the intrinsics from librealsense into OpenCV
            # K_left  = camera_matrix(intrinsics["left"])
            # D_left  = fisheye_distortion(intrinsics["left"])
            # K_right = camera_matrix(intrinsics["right"])
            # D_right = fisheye_distortion(intrinsics["right"])
            # (width, height) = (intrinsics["left"].width, intrinsics["left"].height)

            # Get the relative extrinsics between the left and right camera
            #(R, T) = get_extrinsics(streams["left"], streams["right"])

            (R, T) = self.get_extrinsics(self.left_img, self.right_img)

            # We need to determine what focal length our undistorted images should have
            # in order to set up the camera matrices for initUndistortRectifyMap.  We
            # could use stereoRectify, but here we show how to derive these projection
            # matrices from the calibration and a desired height and field of view

            # We calculate the undistorted focal length:
            #
            #         h
            # -----------------
            #  \      |      /
            #    \    | f  /
            #     \   |   /
            #      \ fov /
            #        \|/
            stereo_fov_rad = 90 * (pi/180)  # 90 degree desired fov
            stereo_height_px = 300          # 300x300 pixel stereo output
            stereo_focal_px = stereo_height_px/2 / tan(stereo_fov_rad/2)

            # We set the left rotation to identity and the right rotation
            # the rotation between the cameras
            R_left = np.eye(3)
            R_right = np.eye(3) #+==========  R

            # The stereo algorithm needs max_disp extra pixels in order to produce valid
            # disparity on the desired output region. This changes the width, but the
            # center of projection should be on the center of the cropped image
            stereo_width_px = stereo_height_px + max_disp
            stereo_size = (stereo_width_px, stereo_height_px)
            stereo_cx = (stereo_height_px - 1)/2 + max_disp
            stereo_cy = (stereo_height_px - 1)/2

            # Construct the left and right projection matrices, the only difference is
            # that the right projection matrix should have a shift along the x axis of
            # baseline*focal_length
            P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                               [0, stereo_focal_px, stereo_cy, 0],
                               [0,               0,         1, 0]])
            P_right = P_left.copy()
            # print("-- T[0][0]: ", T[0][0])
            # print("sf_px: ", stereo_focal_px)

            P_right[0][3] = T[0][0]*stereo_focal_px

            # Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
            # since we will crop the disparity later
            Q = np.array([[1, 0,       0, -(stereo_cx - max_disp)],
                          [0, 1,       0, -stereo_cy],
                          [0, 0,       0, stereo_focal_px],
                          [0, 0, -1/T[0][0], 0]])

            # Create an undistortion map for the left and right camera which applies the
            # rectification and undoes the camera distortion. This only has to be done
            # once
            m1type = cv2.CV_32FC1
            (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(
                K_left, D_left, R_left, P_left, stereo_size, m1type)
            (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(
                K_right, D_right, R_right, P_right, stereo_size, m1type)
            undistort_rectify = {"left": (lm1, lm2),
                                 "right": (rm1, rm2)}

            mode = "stack"
            while True:
                # Check if the camera has acquired any frames
                # frame_mutex.acquire()
                frame_data = {"left": None,
                  "right": None,
                  "timestamp_ms": None
                  }
                # valid = frame_data["timestamp_ms"] is not None
                valid = True
                # frame_mutex.release()

                # If frames are ready to process
                if valid:
                    # Hold the mutex only long enough to copy the stereo frames
                    # frame_mutex.acquire()
                    # frame_copy = {"left": frame_data["left"].copy(),
                    #               "right": frame_data["right"].copy()}
                    # frame_mutex.release()

                    # Undistort and crop the center of the frames
                    center_undistorted = {"left": cv2.remap(src=original_left_img,
                                                            map1=undistort_rectify["left"][0],
                                                            map2=undistort_rectify["left"][1],
                                                            interpolation=cv2.INTER_LINEAR),
                                          "right": cv2.remap(src=original_right_img,
                                                             map1=undistort_rectify["right"][0],
                                                             map2=undistort_rectify["right"][1],
                                                             interpolation=cv2.INTER_LINEAR)}

                    # compute the disparity on the center of the frames and convert it to a pixel disparity (divide by DISP_SCALE=16)
                    disparity = stereo.compute(
                        center_undistorted["left"], center_undistorted["right"]).astype(np.float32) / 16.0

                    # re-crop just the valid part of the disparity
                    disparity = disparity[:, max_disp:]

                    # convert disparity to 0-255 and color it
                    disp_vis = 255*(disparity - min_disp) / num_disp
                    disp_color = cv2.applyColorMap(
                        cv2.convertScaleAbs(disp_vis, 1), cv2.COLORMAP_JET)
                    color_image_l = center_undistorted["left"]
                    print("**CU: ",center_undistorted["left"])
                    # cv2.cvtColor(
                    #     center_undistorted["left"][:, max_disp:], cv2.COLOR_GRAY2RGB)
                    color_image_r = center_undistorted["right"]
                    # cv2.cvtColor(
                    #     center_undistorted["right"][:, max_disp:], cv2.COLOR_GRAY2RGB)
                    # (self.left_img, desired_encoding='passthrough')
                    image_to_publish_l = self.bridge.cv2_to_imgmsg(
                        color_image_l, "bgr8")
                    image_to_publish_r = self.bridge.cv2_to_imgmsg(
                        color_image_r, "bgr8")
                    self.left_img_pub.publish(image_to_publish_l)
                    self.right_img_pub.publish(image_to_publish_r)

                    if mode == "stack":
                        cv2.imshow(WINDOW_TITLE, np.hstack(
                            (color_image_l, disp_color)))
                    if mode == "overlay":
                        ind = disparity >= min_disp
                        color_image_l[ind, 0] = disp_color[ind, 0]
                        color_image_l[ind, 1] = disp_color[ind, 1]
                        color_image_l[ind, 2] = disp_color[ind, 2]
                        cv2.imshow(WINDOW_TITLE, color_image_l)
                key = cv2.waitKey(1)
                if key == ord('s'):
                    mode = "stack"
                if key == ord('o'):
                    mode = "overlay"
                if key == ord('q') or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
        # finally:
        #     print("finally?")


    def shutdown(self):
        rospy.loginfo("Object Detection Test is shutdown")
        rospy.sleep(3)

def main():
    try:
        rospy.init_node('undistort_T265', anonymous=True)
        undistortion_test = Undistort_T265()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

