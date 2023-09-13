# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Displays robust grasps planned using a GQ-CNN-based policy on a set of saved
RGB-D images. The default configuration is cfg/examples/policy.yaml.

Author
------
Jeff Mahler
"""
import argparse
import logging
import numpy as np
import os
import rosgraph.roslogging as rl
import rospy
import sys
import datetime
import threading
import pickle
from matplotlib import pyplot as plt
import cv2

from cv_bridge import CvBridge, CvBridgeError
import ros_numpy
import rosnode
import message_filters
import tf2_ros
from sensor_msgs.msg import Image as ImageMsg, CameraInfo
from geometry_msgs.msg import PointStamped

from autolab_core import (Point, Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage)
from visualization import Visualizer2D as vis

from gqcnn.grasping import Grasp2D, SuctionPoint2D, GraspAction
from gqcnn.msg import GQCNNGrasp
from gqcnn.srv import GQCNNGraspPlanner, GQCNNGraspPlannerSegmask


lock = threading.Lock()

class ImageToGraspPub:
    """Class for a publisher-subscriber listener to work with DexNet

    This class follows a publisher-subscriber model where it subscribes to an
    Image messages consisting of (1) depth image from a top down camera, and a 
    (2) a mask image for the regions of interest as expected by DexNet. The call
    to run_network uses these as input and then publishes the Grasp2D params:
    {(u, v), theta, depth} which correspond to pixel location, grasp angle and depth.

    The topic names for publishing and subscribing are fixed.

    Methods
    -------

    __init__()
        Provide the grasp_estimator object here. Modify other attributes here if
        needed.

    callback_depth(depth_im: Image message)
        Image callback method for ROS. Also registers (using self.step variable) whether 
        a new scene's Image msg is received, using which the Grasp publisher publishes 
        (i.e publish only when a new Image is received).

    run_network()
        Method to perform grasp estimation. Only samples and publishes grasp if the 
        step count variable is updated in callback().
    """

    def __init__(
        self,
        grasp_estimator,
        camera_intr,
        data_staging_loc=None,
        visualize=False,
    ):
        self.estimator = grasp_estimator
        self.camera_intr = camera_intr
        self.visualize = visualize
        self.data_dir = data_staging_loc

        self.depth_im = None
        self.mask_img = None
        self.depth_frame_header = None
        # self.depth_stamp = None
        self.base_frame = "base_link"
        self.SCALING_FACTOR = 1.0
        self.prev_step = None
        self.step = 0  # indicator for whether a new pc is registered

        self.grasp_pub = rospy.Publisher("grasp2D", PointStamped, queue_size=10)
        self.dexnet_vis_pub = rospy.Publisher('/vis/dexnet', ImageMsg, queue_size=10)
        self.pub_rate = rospy.Rate(10)
        # depth_sub = message_filters.Subscriber('/topdown_depth_img', Image, queue_size=10)
        # mask_sub  = message_filters.Subscriber("/topdown_depth_mask", Image, queue_size=10)
        queue_size = 1
        slop_seconds = 0.1
        # ts = message_filters.ApproximateTimeSynchronizer(
        #     [depth_sub, mask_sub], queue_size, slop_seconds
        # )
        # ts.registerCallback(self.callback_depth)

    def callback_depth(self, depth, mask):
        if depth.encoding == '32FC1':
            # depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
            depth_cv = ros_numpy.numpify(depth)
        elif depth.encoding == '16UC1':
            # depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv = ros_numpy.numpify(depth).astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return
        
        mask_img = ros_numpy.numpify(mask)
        print("[CALLBACK] Received depth and mask images!")
        with lock:
            self.depth_im = depth_cv.copy()
            self.mask_img = mask_img.copy()
            self.depth_frame_header = depth.header
            self.step += 1

        # self.run_network()
              

    def run_network(self):
        # New Images are not updated yet!
        # if self.depth_im is None or self.mask_img is None:
        #     return
        if self.prev_step == self.step:
            return # SHOULD NOT HAPPEN!!

        # depth_data = self.depth_im.copy()
        # mask_data = self.mask_img.copy()
        # mask_data = mask_data.astype(np.uint8)
        curr_step = self.step
        if not os.path.exists(os.path.join(self.data_dir, f"step_{curr_step}.txt")):
            return

        print(f"Registered new input data sample for step: {curr_step}!")
        print("Proceeding with grasp sampling")

        with open(os.path.join(self.data_dir, "input_data.pk"), 'rb') as pf:
            input_data = pickle.load(pf)

        depth_data = input_data['depth_img']
        mask_data = input_data['mask_img']
        exp_step = input_data['step']  
       
        print(f"[LISTENER] Setting up network inputs for step:{exp_step}...")
        
        depth_im = DepthImage(depth_data, frame=self.camera_intr.frame)
        segmask  = BinaryImage(mask_data.astype(np.uint8), frame=self.camera_intr.frame)
        color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8), frame=self.camera_intr.frame)

        print("[LISTENER] Predicting Grasps....")
        grasp_response = plan_grasp_segmask(color_im.rosmsg, depth_im.rosmsg, camera_intr.rosmsg, segmask.rosmsg)
        grasp = grasp_response.grasp

        if self.visualize:
            g_center = Point(np.array([grasp.center_px[0], grasp.center_px[1]]),
                       frame=self.camera_intr.frame)
            grasp_2d = Grasp2D(g_center,
                                grasp.angle,
                                grasp.depth,
                                width=0.05,
                                camera_intr=self.camera_intr)
            vis.figure(size=(10, 10))
            vis.imshow(depth_im, vmin=0.6, vmax=0.9)
            vis.grasp(grasp_2d, scale=2.5, show_center=True, show_axis=True)
            vis.title("Planned grasp on depth (Q=%.3f)" % (grasp.q_value))
            vis_filename = os.path.join(self.data_dir, f"output_img_{self.step}.png")
            vis.show(vis_filename)

        # IF VIS_TRUE, THEN ALSO PUBLISH THE IMAGE FILE
        # TODO: LOAD THE FILE FROM DISK 
        dexnet_vis_image = cv2.imread(vis_filename)
        dexnet_vis_image = cv2.cvtColor(dexnet_vis_image, cv2.COLOR_BGR2RGB)
        # TODO: USE this image to publish to rviz here

        # publish the image
        img_msg = ros_numpy.msgify(ImageMsg, dexnet_vis_image, encoding='rgb8')
        self.dexnet_vis_pub.publish(img_msg)
        self.pub_rate.sleep()

        # Publish the Grasps as a PointStamped message
        # Not really publishing!!!
        
        print("[LISTENER] Publishing/Saving...")
        np.savetxt(os.path.join(self.data_dir, f"pub_step_{self.step}.txt"), np.array([self.step]), fmt='%.1e')
        grasp_data = {
            "center" : (grasp.center_px[0], grasp.center_px[1]),
            "angle"  : grasp.angle,
            "z_depth": grasp.depth,
            "q_value": grasp.q_value,
            "step": self.step
        }
        print(grasp_data)
        with open(os.path.join(self.data_dir, f"grasp_data_{self.step}.pk"), 'wb') as pf:
            pickle.dump(grasp_data, pf)

        # ALTERNATIVE: Can also publish the custom Grasp msg after importing it in SceneReplica workspace
        # Construct a Point message: (u, v, theta)
        # grasp_msg = PointStamped()
        # grasp_msg.header = self.depth_frame_header
        # grasp_msg.point.x = grasp.center_px[0]
        # grasp_msg.point.y = grasp.center_px[1]
        # grasp_msg.point.z = grasp.angle
        # while True:
        #     if self.grasp_pub.get_num_connections() > 0:
        #         rospy.loginfo(
        #             f"[LISTENER] Publishing Grasp as Point32 msg!"
        #         )
        #         self.grasp_pub.publish(grasp_msg)
        #         rospy.loginfo("[LISTENER] Finished publishing grasp msg.")
        #         break

        print("[LISTENER] Returning from run_network() call...")
        self.prev_step = self.step
        self.step += 1
        print("=================================================================\n")


def make_parser():
    # Parse args.
    parser = argparse.ArgumentParser(
        description="Run a grasping policy using ROS")

    parser.add_argument("--camera_intr",
                        type=str,
                        default=None,
                        help="path to the camera intrinsics",
                        required=True)
    parser.add_argument("--gripper_width",
                        type=float,
                        default=0.05,
                        help="width of the gripper to plan for")
    parser.add_argument("--namespace",
                        type=str,
                        default="gqcnn",
                        help="namespace of the ROS grasp planning service")
    parser.add_argument("--vis_grasp", 
                        action='store_true',
                        help="whether or not to visualize the grasp")
    return parser
    

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    camera_intr_filename = args.camera_intr
    namespace = args.namespace
    vis_grasp = args.vis_grasp
    EXP_DIR = f"/home/ninad/Desktop/dexnet_exps"

    # Initialize the ROS node.
    rospy.init_node("DexNetGraspingNode")
    # logging.getLogger().addHandler(rl.RosStreamHandler())
    # Wait for grasp planning service and create service proxy.
    rospy.wait_for_service("%s/grasp_planner" % (namespace))
    rospy.wait_for_service("%s/grasp_planner_segmask" % (namespace))
    plan_grasp = rospy.ServiceProxy("%s/grasp_planner" % (namespace),
                                    GQCNNGraspPlanner)
    plan_grasp_segmask = rospy.ServiceProxy(
        "%s/grasp_planner_segmask" % (namespace), GQCNNGraspPlannerSegmask)
    # Set up sensor.
    camera_intr = CameraIntrinsics.load(camera_intr_filename)

    print("[INFO] Setting up the publisher...")
    publisher = ImageToGraspPub(plan_grasp_segmask, camera_intr, EXP_DIR, vis_grasp)

    print("[INFO] Starting the run network loop ...")
    # rospy.spin()
    while not rospy.is_shutdown():
        try:
            publisher.run_network()
        except KeyboardInterrupt:
                break    
    print("[INFO] Exiting DexNet ROS Node")

