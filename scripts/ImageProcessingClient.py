#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import OccupancyGrid
import cv2
from cv_bridge import CvBridge, CvBridgeError
from create_embeeding_grid_map.srv import ImageProcessing, ImageProcessingRequest, MultiImageProcessing, MultiImageProcessingRequest
import numpy as np
import tf2_ros
import message_filters
from sensor_msgs.msg import CameraInfo
# from listener import depth2pc, project_pc_to_image_plane, transform_to_matrix, transform_pc_to_global
from listener import pixel_to_map_coord
from talker import ROSAgentController
from visualization_msgs.msg import Marker, MarkerArray
import json
from openai import OpenAI
import json
import os


directions_name = ['up', 'down', 'left', 'right']


class ImageProcessingClient:
    def __init__(self, init_node=True):
        if init_node:
            rospy.init_node('ImageProcessingClient', anonymous=True)
        
        # 创建 CV Bridge
        self.bridge = CvBridge()

        # 订阅 RGB 图像话题
        # self.rgb_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
        # self.depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
        # self.camera_info_sub = message_filters.Subscriber('/camera/depth/camera_info', CameraInfo)
        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.camera_info_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.images_list = []
        # 等待 ImageProcessing 服务变为可用
        rospy.wait_for_service('image_processing/single')
        rospy.loginfo("Waiting for Single ImageProcessing service...")
        rospy.wait_for_service('image_processing/multiple')
        rospy.loginfo("Connected to Multiple ImageProcessing service.")
        self.image_processing_service = rospy.ServiceProxy('image_processing/single', ImageProcessing)
        self.multi_image_processing_service = rospy.ServiceProxy('image_processing/multiple', MultiImageProcessing)
        rospy.loginfo("Connected to ImageProcessing service.")

        # 用于tf监听器
        self.tf_cache_duration = 2.0
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.tf_cache_duration))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 时间同步器
        ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.camera_info_sub], 10, 0.5)
        ts.registerCallback(self.callback)
        self.agent_controller = ROSAgentController(init_node=False)
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)

        self.openai_key = os.environ['OPENAI_API_KEY']


    def gpt_filter_out_not_object(self, tags_list):
        client = OpenAI()
        # tags_list = tags.split(',')
        # 手动添加地板
        tags_list.append('ceiling')
        tags_list.append('floor')
        tags_list_str = json.dumps(tags_list)
        prompt = f"""You are a professional filter with advanced capabilities, trained to identify and remove non-object names and synonyms from a list of tags. Your task is to analyze each list provided and filter out any names that are not physical objects or are synonyms of other items in the list. Here are two examples to guide you:
Q:["flower", "blossom", "sunshine", "flora", "happiness", "sofa", "couch", "glee", "settee", "joy", "divan"]
A:["flower", "sofa"]
Q:["bed", "bedcover", "bedroom", "blue", "wall", "floor", "picture frame", "pillow", "room", "white"]
A:["bed", "bedcover", "wall", "floor", "picture frame", "pillow"]
Q:{tags_list_str}
A:
"""
        response = client.completions.create(
                        model="gpt-3.5-turbo-instruct-0914",
                        prompt=prompt,
                        temperature=0.1,
                        max_tokens=256,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
        output = (response.choices[0].text).strip()
        result = json.loads(output)
        return result
    
    def callback(self, rgb_msg, depth_msg, camera_info_msg):
        # 将 ROS Image 消息转换为 OpenCV 图像
        try:
            camera_matrix = np.array(camera_info_msg.K).reshape((3, 3))
            transform = self.tf_buffer.lookup_transform('map', 'camera_depth_optical_frame', rospy.Time(0), rospy.Duration(0.1))
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
            cv_rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            self.images_list.append((cv_rgb_image,cv_depth_image))
            
            # Convert Depth Image to PointCloud
            # pointcloud_local, _ = depth2pc(depth=cv_depth_image, cam_mat=camera_matrix)
            # transform_map_frame_matrix = transform_to_matrix(transform)
            # pointcloud_global = transform_pc_to_global(pointcloud_local, transform_map_frame_matrix)

        except CvBridgeError as e:
            print(e)
            return
        except Exception as e:
            print(e)
            return
        if len(self.images_list) == 4:
            try:
                rospy.loginfo("Calling ImageProcessing service...")
                service_request = MultiImageProcessingRequest()
                service_request.images = [self.bridge.cv2_to_imgmsg(image, "bgr8") for image,_ in self.images_list]
                service_request.target = 'find_a_kichen'
                depth_images = [depth for _,depth in self.images_list]
                response = self.multi_image_processing_service(service_request)
                self.images_list = []
                pred_phrases = response.pred_phrases
                bounding_boxes = np.array(response.bounding_boxes)
                bounding_boxes = bounding_boxes.reshape((-1,4))
                img_num = len(pred_phrases)
                bounding_boxes_per_image_per_phrase = {}
                all_object_names = []
                for i in range(img_num):
                    bounding_boxes_per_image_per_phrase[i] = {}
                    for j, phrase in enumerate(pred_phrases[i].split(',')):
                        bounding_boxes_per_image_per_phrase[i][phrase] = bounding_boxes[i*4+j]
                        all_object_names.append(phrase)
                img_phrase_box_dict = {}
                for i in range(img_num):
                    img_phrase_box_dict[i] = {}
                    for j, phrase in enumerate(pred_phrases[i].split(',')):
                        map_coord = pixel_to_map_coord(pixel_x=int((bounding_boxes_per_image_per_phrase[i][phrase][0]+bounding_boxes_per_image_per_phrase[i][phrase][2])/2), pixel_y=int((bounding_boxes_per_image_per_phrase[i][phrase][1]+bounding_boxes_per_image_per_phrase[i][phrase][3])/2), depth_image=depth_images[i], camera_matrix=camera_matrix, camera_pose=transform)
                        # skip the object that contain nan coord
                        if np.isnan(map_coord).any():
                            continue
                        img_phrase_box_dict[i][phrase] = {}
                        img_phrase_box_dict[i][phrase]['bbox'] = bounding_boxes_per_image_per_phrase[i][phrase]
                        img_phrase_box_dict[i][phrase]['map_coord'] = map_coord
                rospy.loginfo(f"Image processed. Reulst:{img_phrase_box_dict}")
                # Using GPT to Filter out non-object
                rospy.loginfo(f"All Object Names: {all_object_names}")
                rospy.loginfo(f"GPT Filtering out non-object names...")
                filtered_object_names = self.gpt_filter_out_not_object(all_object_names)
                rospy.loginfo(f"Filtered Object Names: {filtered_object_names}")
                filtered_img_phrase_box_dict = {}
                for i in range(img_num):
                    filtered_img_phrase_box_dict[i] = {}
                    for j, phrase in enumerate(pred_phrases[i].split(',')):
                        if phrase in filtered_object_names:
                            filtered_img_phrase_box_dict[i][phrase] = img_phrase_box_dict[i][phrase]
                # Random Choose one object
                random_img_id = np.random.choice(list(img_phrase_box_dict.keys()))
                random_phrase = np.random.choice(list(img_phrase_box_dict[random_img_id].keys()))
                rospy.loginfo(f"Randomly Selected Object: {random_phrase}")
                # Get the map coordinate of the selected object
                selected_obj_map_coord = img_phrase_box_dict[random_img_id][random_phrase]['map_coord']
                # Create a MarkerArray message
                marker_array = MarkerArray()
                marker_id = 0
                for img_id, phrase_box_dict in img_phrase_box_dict.items():
                    # Create a marker for each object
                    for phrase, box_map_coord_dict in phrase_box_dict.items():
                        marker = Marker()
                        marker.header.frame_id = "map"
                        marker.header.stamp = rospy.Time.now()
                        marker.ns = "objects"
                        marker.id = marker_id
                        marker_id += 1
                        marker.type = Marker.TEXT_VIEW_FACING
                        marker.action = Marker.ADD
                        marker.pose.position.x = box_map_coord_dict['map_coord'][0]
                        marker.pose.position.y = box_map_coord_dict['map_coord'][1]
                        marker.pose.position.z = box_map_coord_dict['map_coord'][2]  # Adjust the Z position if needed
                        marker.pose.orientation.w = 1.0
                        marker.text = phrase
                        marker.scale.z = 0.5  # Text size
                        marker.color.a = 1.0  # Don't forget to set the alpha!
                        marker.color.r = 1.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                        marker_array.markers.append(marker)
                        marker = Marker()
                        marker.header.frame_id = "map"
                        marker.header.stamp = rospy.Time.now()
                        marker.ns = "objects"
                        marker.id = marker_id
                        marker_id += 1
                        marker.type = Marker.SPHERE  # Change type to SPHERE
                        marker.action = Marker.ADD
                        marker.pose.position.x = box_map_coord_dict['map_coord'][0]
                        marker.pose.position.y = box_map_coord_dict['map_coord'][1]
                        marker.pose.position.z = box_map_coord_dict['map_coord'][2]  # Adjust the Z position if needed
                        marker.pose.orientation.w = 1.0
                        marker.scale.x = 0.5  # Sphere diameter in the X direction
                        marker.scale.y = 0.5  # Sphere diameter in the Y direction
                        marker.scale.z = 0.5  # Sphere diameter in the Z direction
                        marker.scale.z = 0.5  # Text size
                        marker.color.a = 1.0  # Don't forget to set the alpha!
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                        marker_array.markers.append(marker)
                # Publish the MarkerArray
                self.marker_pub.publish(marker_array)

                # Goto the selected object
                # self.agent_controller.nav2point(selected_obj_map_coord[0], selected_obj_map_coord[1])
      
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)

        # # 调用图像处理服务
        # try:
        #     rospy.loginfo("Calling ImageProcessing service...")
        #     service_request = ImageProcessingRequest()
        #     service_request.image = self.bridge.cv2_to_imgmsg(cv_rgb_image, "bgr8")
        #     response = self.image_processing_service(service_request)
        #     rospy.loginfo("Image processed. Tags: %s, Tags Chinese: %s" % (response.tags, response.tags_chinese))
        # except rospy.ServiceException as e:
        #     rospy.logerr("Service call failed: %s" % e)

if __name__ == '__main__':
    image_processing_client = ImageProcessingClient()
    rospy.spin()