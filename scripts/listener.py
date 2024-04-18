#!/usr/bin/env python
import torch
import rospy
import message_filters
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
import cv2
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import geometry_msgs.msg
import numpy as np
import tf
from geometry_msgs.msg import TransformStamped
import tf2_geometry_msgs
import std_msgs
from sensor_msgs.msg import CameraInfo
import sensor_msgs.point_cloud2 as pc2


def transform_pc_to_global(pc, pose):
    """
    Transform a point cloud (pc) given a camera pose.
    This function is robust to NaN values in the point cloud.

    :param pc: numpy array representing the point cloud, may contain NaN values.
    :param pose: the pose of the camera coordinate in which the pc is.
    :return: Transformed point cloud with the same shape as input.
    """
    # Create a mask of NaN values
    nan_mask = np.isnan(pc)

    # Replace NaNs with zeros for calculation
    pc_safe = np.where(nan_mask, 0, pc)

    # Transform the pc_safe to homogeneous coordinates
    pc_homo = np.vstack([pc_safe, np.ones((1, pc_safe.shape[1]))])

    # Apply the transformation
    pc_global_homo = pose @ pc_homo

    # Convert back to non-homogeneous coordinates
    transformed_pc = pc_global_homo[:3, :]

    # Re-apply NaNs to maintain the original structure
    transformed_pc[nan_mask] = np.nan

    return transformed_pc


def transform_to_matrix(transform_stamped):
    """
    Convert a TransformStamped object to a 4x4 homogeneous transformation matrix.

    Args:
    transform_stamped (TransformStamped): A TransformStamped object as provided by tf2_ros. 
    This object includes translation and rotation information from one frame to another.

    Returns:
    numpy.ndarray: A 4x4 homogeneous transformation matrix representing the rotation and translation
    defined in the TransformStamped object.
    """
    # Extract translation and rotation from the transform
    trans = transform_stamped.transform.translation
    rot = transform_stamped.transform.rotation

    # Convert to a matrix (4x4) using tf
    translation = [trans.x, trans.y, trans.z]
    rotation = [rot.x, rot.y, rot.z, rot.w]
    matrix = tf.transformations.quaternion_matrix(rotation)
    matrix[:3, 3] = translation
    return matrix


def depth2pc_realworld(depth, cam_mat):
    """
    Converts a depth image to a 3D point cloud in real-world coordinates.

    This function projects each pixel in the depth image into 3D space based on the depth value and the inverse of the camera matrix. It also applies a mask to filter out points based on their depth values (z-coordinate), keeping only points within a specific range.

    Args:
    depth (numpy.ndarray): A 2D array representing the depth image, where each element is a depth value.
    cam_mat (numpy.ndarray): The camera intrinsic matrix (3x3) representing the camera's internal parameters.

    Returns:
    tuple:
        - numpy.ndarray: A 3xN array where N is the number of pixels in the depth image. Each column represents a 3D point in real-world coordinates.
        - numpy.ndarray: A boolean mask array indicating which points are within the specified depth range.
    """

    h, w = depth.shape
    cam_mat_inv = np.linalg.inv(cam_mat)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    x = x.reshape((1, -1))[:, :]
    y = y.reshape((1, -1))[:, :]
    z = depth.reshape((1, -1))[:, :]

    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = cam_mat_inv @ p_2d
    pc = pc * z
    mask_1 = pc[2, :] > 0.1
    mask_2 = pc[2, :] < 4
    mask = np.logical_and(mask_1, mask_2)
    return pc, mask


def depth2pc(depth, cam_mat):
    """
    Converts a depth image to a 3D point cloud without applying any real-world range constraints.

    This function projects each pixel in the depth image into 3D space using the depth value and the inverse of the camera matrix. Unlike depth2pc_realworld, this function does not apply a depth range filter, and hence, all converted points are included.

    Args:
    depth (numpy.ndarray): A 2D array representing the depth image, where each element is a depth value.
    cam_mat (numpy.ndarray): The camera intrinsic matrix (3x3) representing the camera's internal parameters.

    Returns:
    tuple:
        - numpy.ndarray: A 3xN array where N is the number of pixels in the depth image. Each column represents a 3D point in the camera coordinate system.
        - numpy.ndarray: A boolean mask array indicating which points have a depth value greater than a specified threshold (in this case, 0.1 meters).
    """

    h, w = depth.shape
    cam_mat_inv = np.linalg.inv(cam_mat)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    x = x.reshape((1, -1))[:, :]
    y = y.reshape((1, -1))[:, :]
    z = depth.reshape((1, -1))[:, :]

    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = cam_mat_inv @ p_2d
    pc = pc * z
    mask_1 = pc[2, :] > 0.1
    return pc, mask_1


def pos2grid(pos, map_msg):
    """
    Convert a position in the map coordinate system to grid coordinates.

    Args:
    pos (tuple): A tuple (x, y) representing the position in the map coordinate system.
    map_msg (nav_msgs/OccupancyGrid): The occupancy grid data which includes resolution and origin.

    Returns:
    tuple: A tuple (grid_x, grid_y) representing the position in the grid coordinates.
    """
    # Extract resolution and origin information from map_msg
    resolution = map_msg.info.resolution
    origin = map_msg.info.origin.position

    # Convert points from the map coordinate system to grid coordinates
    grid_x = int((pos[0] - origin.x) / resolution)
    grid_y = int((pos[1] - origin.y) / resolution)

    return grid_x, grid_y

def grid2pos(grid_x, grid_y, map_msg):
    """
    Convert grid coordinates to a position in the map coordinate system.

    Args:
    grid_x (int): The x-coordinate in the grid.
    grid_y (int): The y-coordinate in the grid.
    map_msg (nav_msgs/OccupancyGrid): The occupancy grid data which includes resolution and origin.

    Returns:
    tuple: A tuple (x, y) representing the position in the map coordinate system.
    """
    # Extract resolution and origin from map_msg
    resolution = map_msg.info.resolution
    origin = map_msg.info.origin.position

    # Convert grid coordinates to map coordinates
    x = (grid_x * resolution) + origin.x
    y = (grid_y * resolution) + origin.y

    return x, y


def pc2grid(pc, map_msg):
    """
    Convert a pointcloud in space to grid coordinates in the OccupancyGrid,
    while also returning the positions of NaN values in the point cloud.

    Args:
    pc (numpy.ndarray): A 3xN array of N points in space, may contain NaN values.
    map_msg (OccupancyGrid): The OccupancyGrid message which provides metadata about the grid.

    Returns:
    tuple:
        - numpy.ndarray: A 2xM array of grid coordinates corresponding to the M non-NaN input points.
        - numpy.ndarray: An array of indices of the NaN points in the original point cloud.
    """

    # Extract the map's resolution and origin
    resolution = map_msg.info.resolution
    origin = map_msg.info.origin.position

    # Identify non-NaN points
    valid_mask = ~np.isnan(pc[0, :]) & ~np.isnan(pc[1, :])
    valid_pc = pc[:, valid_mask]

    assert not np.isnan(valid_pc).any(), "NaN values detected in valid_pc"

    # Calculate positions relative to the map origin
    relative_x = valid_pc[0, :] - origin.x
    relative_y = valid_pc[1, :] - origin.y

    # Convert positions to grid coordinates
    grid_x = np.floor(relative_x / resolution).astype(int)
    grid_y = np.floor(relative_y / resolution).astype(int)

    # Get indices of NaN values
    nan_indices = np.where(~valid_mask)[0]

    return (np.vstack([grid_x, grid_y]), nan_indices)


def grid2pc(grid_coords, map_msg):
    """
    Convert grid coordinates in the OccupancyGrid to point cloud coordinates in space.

    Args:
    grid_coords (numpy.ndarray): A 2xM array of M grid coordinates.
    map_msg (OccupancyGrid): The OccupancyGrid message which provides metadata about the grid.

    Returns:
    numpy.ndarray: A 3xM array of M points in space corresponding to the input grid coordinates.
    """

    # Extract the map's resolution and origin
    resolution = map_msg.info.resolution
    origin = map_msg.info.origin.position

    # Convert grid coordinates to positions relative to the map origin
    relative_x = (grid_coords[0, :] * resolution) + origin.x
    relative_y = (grid_coords[1, :] * resolution) + origin.y

    # Form the point cloud coordinates, z is set to all 1
    pc = np.vstack([relative_x, relative_y, np.ones_like(relative_x)])

    return pc


def pixel_to_map_coord(pixel_x, pixel_y, depth_image, camera_matrix, camera_pose):
    """
    Convert pixel coordinates in an RGB image to map coordinates.

    Args:
        pixel_x (int): The x-coordinate of the pixel in the RGB image.
        pixel_y (int): The y-coordinate of the pixel in the RGB image.
        depth_image (numpy.ndarray): The depth image aligned with the RGB image.
        camera_matrix (numpy.ndarray): The camera intrinsic matrix (3x3).
        camera_pose (geometry_msgs.msg.TransformStamped): The camera's pose in the map frame.
        map_msg (nav_msgs.msg.OccupancyGrid): The map data.

    Returns:
        tuple: A tuple (grid_x, grid_y) representing the position in the grid coordinates.
    """
    # Get depth for the pixel
    depth = depth_image[pixel_y, pixel_x]

    # Convert pixel to 3D point in camera coordinates
    point_2d = np.array([pixel_x, pixel_y, 1])
    cam_point_3d = depth * np.linalg.inv(camera_matrix) @ point_2d

    # Convert camera coordinates to global (map) frame
    cam_point_3d_homogenous = np.append(cam_point_3d, 1)
    global_point_3d = transform_to_matrix(camera_pose) @ cam_point_3d_homogenous

    # x, y = global_point_3d[:2]
    # Convert global coordinates to grid coordinates
    # grid_x, grid_y = pos2grid((global_point_3d[0], global_point_3d[1]), map_msg)
    return global_point_3d


def project_point(cam_mat, p):
    """
    Projects a 3D point in camera coordinates onto the 2D image plane.

    Args:
    cam_mat (numpy.ndarray): The camera intrinsic matrix (3x3) representing the camera's internal parameters.
    p (numpy.ndarray): A 3D point in camera coordinates (x, y, z).

    Returns:
    tuple: A tuple (x, y, z) where x and y are the coordinates of the projected point on the 2D image plane,
           and z is the depth value (distance from the camera).

    The function operates as follows:
    1. Reshape the input point `p` to a 3x1 column vector, if not already in that shape.
    2. Multiply the camera matrix `cam_mat` with the reshaped point `p` to get the point in homogeneous image coordinates.
    3. Normalize the resulting point by its third element (z-coordinate) to convert from homogeneous coordinates
       to Cartesian coordinates on the image plane.
    4. The x and y values are then rounded to the nearest integer to get pixel indices, with an added 0.5
       to ensure correct rounding behavior. This is because pixel indices are integers.
    5. The original z value (depth) is preserved and returned along with the x and y pixel coordinates.
    """
    new_p = cam_mat @ p.reshape((3, 1))
    z = new_p[2, 0]
    new_p = new_p / new_p[2, 0]
    x = int(new_p[0, 0] + 0.5)
    y = int(new_p[1, 0] + 0.5)
    return x, y, z


def project_pc_to_image_plane(cam_mat, pc):
    """
    Projects each point in a 3D point cloud onto the 2D image plane using a camera matrix.
    Handles NaN values in the point cloud and retains their indices.

    Args:
    cam_mat (numpy.ndarray): The camera intrinsic matrix (3x3) representing the camera's internal parameters.
    pc (numpy.ndarray): A point cloud, an array of 3D points in camera coordinates (3xN), where N is the number of points.

    Returns:
    tuple:
        - numpy.ndarray: An array of 2D coordinates of the projected points on the image plane (2xM), where M is the number of non-NaN points.
        - numpy.ndarray: An array of indices of the NaN points in the original point cloud.
    """
    # Identify NaN values and create a mask
    nan_mask = np.isnan(pc[0, :]) | np.isnan(pc[1, :]) | np.isnan(pc[2, :])
    valid_pc = pc[:, ~nan_mask]

    # Project points onto the image plane
    projected_pc_homo = cam_mat @ valid_pc
    projected_pc_homo /= projected_pc_homo[2, :]

    # Convert to pixel indices
    projected_pc_2d = projected_pc_homo[:2, :].astype(int)

    # Get indices of NaN values
    nan_indices = np.where(nan_mask)[0]

    return projected_pc_2d, nan_indices


class RGBDMapSync:
    def __init__(self):
        rospy.init_node('rgbd_map_sync', anonymous=True)
        
        # 创建 CV Bridge
        self.bridge = CvBridge()

        # 初始化 tf2 缓冲区和监听器
        self.tf_cache_duration = 2.0
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.tf_cache_duration))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 订阅 RGB 和深度图像话题以及地图话题
        # rgb_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
        # depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
        # camera_info_sub = message_filters.Subscriber('/camera/depth/camera_info', CameraInfo)
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        camera_info_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        map_sub = message_filters.Subscriber('/map', OccupancyGrid)

        # 时间同步器
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, map_sub, camera_info_sub], 10, 0.5)
        ts.registerCallback(self.callback)
        

        # 创建一个发布者来发布图像
        self.image_pub = rospy.Publisher("/map_visualization", Image, queue_size=1)

        print("Time Synchronizer registered")

    def callback(self, rgb_msg, depth_msg, map_msg, aligned_cam_info_msg):
        print("Received RGB and Depth images and map data")
        time_stamp = rgb_msg.header.stamp
        try:
            transform = self.tf_buffer.lookup_transform('map', 'camera_depth_optical_frame', rospy.Time(0), rospy.Duration(0.1))
            # 将 ROS 图像消息转换为 OpenCV 图像
            cv_rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
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
            # 处理图像和地图数据
            self.process_data(cv_rgb_image, cv_depth_image, map_msg, transform, aligned_cam_info_msg)

        except CvBridgeError as e:
            rospy.logerr(e)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to get transform: %s" % e)
        except Exception as e:
            rospy.logerr("An error occurred: %s" % e)

    def process_data(self, rgb_image, depth_image, map_msg, transform, aligned_cam_info_msg):
        # Recieved Intrinsic depth camera matrix
        # Assume rgb_image is already aligned with depth_image
        # For RealSense Camera D435, we will use /camera/camera/aligned_depth_to_color/camera_info and /camera/camera/aligned_depth_to_color/image_raw
        # For Gazebo Turtlebo3 Waffle, it's already aligned
        camera_matrix = np.array(aligned_cam_info_msg.K).reshape((3, 3))

        # Camera Position in Map Frame
        camera_pose_x = transform.transform.translation.x
        camera_pose_y = transform.transform.translation.y
        camera_grid_index_y, camera_grid_index_x = pos2grid((camera_pose_x,camera_pose_y), map_msg=map_msg)
        camera_grid_pos = (camera_grid_index_x, camera_grid_index_y)
        # Convert Depth Image to PointCloud
        pointcloud_local, _ = depth2pc(depth=depth_image, cam_mat=camera_matrix)
        # Transform all points to the global frame
        transform_map_frame_matrix = transform_to_matrix(transform)
        pointcloud_global = transform_pc_to_global(pointcloud_local, transform_map_frame_matrix)
        # self.map_visualize(camera_grid_pos, rgb_image, pointcloud_local, pointcloud_global, map_msg, camera_matrix)

    
    def map_visualize(self, camera_grid_pos, rgb_image, pointcloud_local, pointcloud_global, map_msg, camera_matrix):
        map_width = map_msg.info.width
        map_height = map_msg.info.height
        camera_grid_index_y, camera_grid_index_x = camera_grid_pos
        # Transform all global points to indexes of nav_msgs/OccupancyGrid
        grid_ids, nan_indices = pc2grid(pc=pointcloud_global, map_msg=map_msg)
        # ---map可视化---
        
        # Try to project all points back to rgb plane to test if the transformation is correct
        rgb_indices, nan_indices_2 = project_pc_to_image_plane(cam_mat=camera_matrix, pc=pointcloud_local)
        # ignore all the points out of rgb_image range
        x_indices, y_indices = rgb_indices
        valid_x_indices = x_indices[(x_indices >= 0) & (x_indices < rgb_image.shape[1])]
        valid_y_indices = y_indices[(y_indices >= 0) & (y_indices < rgb_image.shape[0])]
        
        occupancygrid_map = np.array(map_msg.data).reshape((map_height, map_width))
        # selected_grids_in_map = occupancygrid_map[grid_ids[1, :], grid_ids[0, :]]
        # occupancygrid_map[grid_ids[1, :], grid_ids[0, :]] = 
        map_visual = np.zeros((map_height, map_width, 3), dtype=np.uint8)
        free_area = occupancygrid_map == 0
        occupied_area = occupancygrid_map == 100
        unknown_area = occupancygrid_map == -1
        # 空闲区为白色
        map_visual[free_area] = [255, 255, 255]
        # 占用区为灰色
        map_visual[occupied_area] = [128, 128, 128]
        # 未知区为浅灰
        map_visual[unknown_area] = [50, 50, 50]
        # 点云映射区为红色
        # map_visual[grid_ids[1, :], grid_ids[0, :]] = [0, 0, 255]

        # grid映射到rgb
        map_visual[grid_ids[1, :], grid_ids[0, :]] = [0,0,0]
        map_visual[grid_ids[1, :], grid_ids[0, :]] = rgb_image[valid_y_indices, valid_x_indices]
        
        # 相机所在位置画绿色圆
        cv2.circle(map_visual, (camera_grid_index_y, camera_grid_index_x), 5, (0, 255, 0), -1)
        try:
            ros_image = self.bridge.cv2_to_imgmsg(map_visual, "bgr8")
            # ros_image = self.bridge.cv2_to_imgmsg(projected_rgb_plane, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        # 发布图像
        self.image_pub.publish(ros_image)
         

if __name__ == '__main__':
    rgbd_map_sync = RGBDMapSync()
    rospy.spin()
