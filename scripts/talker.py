#!/usr/bin/env python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import Image
import tf
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
from collections import deque
import tf2_ros
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import message_filters
from visualization_msgs.msg import Marker, MarkerArray
import threading
from geometry_msgs.msg import PointStamped
import numpy as np


class ROSAgentController:
    def __init__(self, init_node=True):
        if init_node:
            rospy.init_node('agent_controller', anonymous=True)
        rospy.loginfo("[AgentController]:Waiting for move_base action server...")
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        rospy.loginfo("[AgentController]:Done")
        self.bridge = CvBridge()
        self.observations = deque(maxlen=4)
        self.lock = threading.Lock()
        # 订阅RGB和深度图像的主题
        # self.rgb_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback)
        # self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        # self.camera_info_sub = rospy.Subscriber("/camera/depth/camera_info", Image, self.depth_callback)
        # rgb_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
        # depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
        # camera_info_sub = message_filters.Subscriber('/camera/depth/camera_info', CameraInfo)
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        camera_info_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.frontier_sub = rospy.Subscriber('/explore/frontiers', MarkerArray, self.frontiers_callback)
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)

        # 初始化 tf2 缓冲区和监听器
        self.tf_cache_duration = 2.0
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.tf_cache_duration))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 时间同步器
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, camera_info_sub], 10, 0.5)
        ts.registerCallback(self.callback)

        # 用于tf监听器
        self.tf_cache_duration = 2.0
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.tf_cache_duration))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.transform = None
        self.frontiers_msg = None
        self.frontiers = None

    def get_current_position(self, return_transform=False):
        try:
            # 获取当前时刻base_link相对于map的变换
            transform = self.tf_buffer.lookup_transform(target_frame='map', source_frame='base_link',
                                                        time=rospy.Time(0), timeout=rospy.Duration(0.5))
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            if return_transform:
                return transform
            else:
                return translation, rotation
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to get transform: %s" % e)
        except Exception as e:
            print(e)
            return None, None

    def nav2point(self, map_x, map_y, wait_for_result=True):
        # 创建MoveBaseGoal消息
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        # 设置目标位置
        goal.target_pose.pose.position.x = map_x
        goal.target_pose.pose.position.y = map_y
        goal.target_pose.pose.orientation.w = 1.0  # 有效的默认方向

        # 发送目标
        self.client.send_goal(goal)
        if wait_for_result:
            wait = self.client.wait_for_result()
            if not wait:
                rospy.logerr("Action server not available!")
                return None
            else:
                return self.client.get_result()
        else:
            return self.client

    def rotate_angle(self, angle_in_degrees):
        # 获取当前位置
        current_transition, _ = self.get_current_position()
        if current_transition is None:
            return None
        current_position = [current_transition.x, current_transition.y, current_transition.z]

        # 将角度转换为四元数
        quaternion = tf.transformations.quaternion_from_euler(0, 0, math.radians(angle_in_degrees))

        # 创建MoveBaseGoal消息
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        # 设置目标位置和方向
        goal.target_pose.pose.position.x = current_position[0]
        goal.target_pose.pose.position.y = current_position[1]
        goal.target_pose.pose.position.z = current_position[2]
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]

        # 发布目标
        self.client.send_goal(goal)
        wait = self.client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available!")
        else:
            result = self.client.get_result()
            self.capture_images()
            return result

    def rotate_local_angle(self, angle_in_degrees):
        # 将角度转换为四元数
        quaternion = tf.transformations.quaternion_from_euler(0, 0, math.radians(angle_in_degrees))

        # 创建MoveBaseGoal消息
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "base_link"
        goal.target_pose.header.stamp = rospy.Time.now()

        # 设置目标方向
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]

        # 发布目标
        self.client.send_goal(goal)
        wait = self.client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available!")
        else:
            return self.client.get_result()

    def rotate_angle_by_duration(self, angular_speed, duration):
        # 创建一个用于发送速度命令的发布者
        cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.sleep(1)  # 等待连接

        # 创建Twist消息，用于控制角速度
        twist = Twist()
        twist.angular.z = angular_speed  # 设置角速度

        # 计算结束时间
        end_time = rospy.Time.now() + rospy.Duration(duration)

        # 发布速度命令，直到达到指定的时间
        while rospy.Time.now() < end_time:
            cmd_vel_publisher.publish(twist)
            rospy.sleep(0.1)  # 小延时以避免过载
            self.capture_images()

        # 停止机器人
        twist.angular.z = 0
        cmd_vel_publisher.publish(twist)

    def frontiers_callback(self, frontiers_marker_array_msg):
        # frontiers = []
        # for marker in frontiers_marker_array_msg.markers:
        # frontiers.append([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
        self.frontiers_msg = frontiers_marker_array_msg

    def callback(self, rgb_msg, depth_msg, camera_info):
        try:
            with self.lock:
                self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                # 判断深度图像的格式并进行相应处理
                if depth_msg.encoding == '16UC1':
                    # 如果是16UC1格式，转换为32FC1
                    depth_image_raw = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                    depth_image_float = np.array(depth_image_raw, dtype=np.float32)  # 转换为浮点数
                    depth_image_float = depth_image_float / 1000.0  # 从毫米转换为米
                    self.depth_image = depth_image_float
                elif depth_msg.encoding == '32FC1':
                    # 如果已经是32FC1格式，直接使用
                    self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
                else:
                    rospy.logerr("Unsupported depth image format: %s" % depth_msg.encoding)
                    return


                self.transform = self.tf_buffer.lookup_transform('map', 'camera_depth_optical_frame', rospy.Time(0),
                                                                 rospy.Duration(0.5))
                self.camera_info = camera_info
        except CvBridgeError as e:
            rospy.logerr("Failed to convert image: %s" % e)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to get transform: %s" % e)
        except Exception as e:
            print(e)
            rospy.logerr("Callback Failed to run %s" % e)

    def capture_images(self):
        # 确保收到图像
        while self.rgb_image is None or self.depth_image is None:
            rospy.sleep(0.1)
        with self.lock:
            self.observations.append({"rgb": self.rgb_image, "depth": self.depth_image, "camera_info": self.camera_info,
                                      "transform": self.transform})

    def get_frontier(self):
        # 确保收到图像
        while self.frontiers_msg is None:
            rospy.sleep(0.1)
        with self.lock:
            frontiers = []
            for marker in self.frontiers_msg.markers:
                frontiers.append([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
            self.frontiers = frontiers
        return frontiers


if __name__ == '__main__':
    agent_controller = ROSAgentController()

    for i in range(4):
        agent_controller.rotate_angle(i * 90)
    # # 从队列中获取图像并显示
    # for observation in agent_controller.observations:
    #     cv2.imshow("RGB", observation["rgb"])
    #     cv2.imshow("Depth", observation["depth"])
    #     cv2.waitKey(0)
