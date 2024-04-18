#!/usr/bin/env python
from __future__ import print_function
from create_embeeding_grid_map.srv import ProcessText, ProcessTextResponse, GetState, GetStateResponse
import rospy
from dataclasses import dataclass
import random
import threading
import time
from talker import ROSAgentController
from dataclasses import dataclass

from create_embeeding_grid_map.srv import ImageProcessing, ImageProcessingRequest, MultiImageProcessing, \
    MultiImageProcessingRequest
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import numpy as np
from listener import pixel_to_map_coord
from visualization_msgs.msg import Marker, MarkerArray
from openai import OpenAI, APIConnectionError
import json
import os
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String



class StateBase(object):
    def __init__(self, id, name, detail, state_machine):
        self.id = id
        self.name = name
        self.detail = detail
        self.state_machine = state_machine

    def process_state(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def perform_action(self):
        raise NotImplementedError("This method should be overridden by subclasses")


class LookSurroundingState(StateBase):
    def process_state(self):
        rospy.logdebug("Look Surrounding Processing")
        success = self.perform_action()  # Placeholder for actual logic
        if success:
            rospy.logdebug("Look Surrounding: Success, Transition to ObjectDetection")
            self.state_machine.try_set_state_by_name("object_detect")
            return ProcessTextResponse("Look Surrounding successful")
        else:
            rospy.logwarn("Look Surrounding: Failed, Transition to Idle")
            self.state_machine.try_set_state_by_name("idle")
            # TODO: Log Reason for failure
            return ProcessTextResponse("Look Surrounding failed")

    def perform_action(self):
        self.state_machine.status_pub.publish("Robot Look Surrounding...")
        rospy.logdebug("Look Surrounding Performing")
        try:
            for i in range(4):
                # 每90度旋转一次
                self.state_machine.agent_controller.rotate_angle(i * 90)
            return True
        except Exception as e:
            rospy.logerr(f"Exception in perform_look_surrounding: {e}")
            return False


class ObjectDetectionState(StateBase):
    def process_state(self):
        rospy.logdebug("ObjectDetection Processing")
        success = self.perform_action()  # Placeholder for actual logic
        if success:
            rospy.logdebug("ObjectDetection: Success, Transition to GPTChooseObject")
            self.state_machine.try_set_state_by_name("gpt_choose_object")
            return ProcessTextResponse("Surrounding ObjectDetection successful")
        else:
            if len(self.state_machine.object_decision['result_dict']) == 0:
                rospy.logwarn("ObjectDetection: Failed, No valid object detected, Transition to NearestFrontier")
                self.state_machine.try_set_state_by_name("nearest_frontier")
                return ProcessTextResponse("Surrounding ObjectDetection failed")

            rospy.logwarn("ObjectDetection: Failed, Transition to Idle")
            self.state_machine.try_set_state_by_name("idle")
            # TODO: Log Reason for failure
            return ProcessTextResponse("Surrounding ObjectDetection failed")

    def perform_action(self):
        self.state_machine.status_pub.publish("Surrounding Images Saved, Send to Detect Objects...")
        rospy.logdebug("Surrounding ObjectDetection  Performing")
        try:
            self.state_machine.object_detection_bounding_boxes = []
            observations = []
            while self.state_machine.agent_controller.observations:
                observations.append(self.state_machine.agent_controller.observations.pop())
            rgb_images = [obs['rgb'] for obs in observations]
            depth_images = [obs['depth'] for obs in observations]
            camera_matrix = np.array(observations[0]['camera_info'].K).reshape((3, 3))
            transforms = [obs['transform'] for obs in observations]
            service_request = MultiImageProcessingRequest()
            service_request.images = [self.state_machine.bridge.cv2_to_imgmsg(image, "bgr8") for image in rgb_images]
            service_request.target = self.state_machine.target.replace(' ', '_')
            response = self.state_machine.multi_image_processing_service_proxy(service_request)
            pred_phrases = response.pred_phrases
            bounding_boxes = np.array(response.bounding_boxes).reshape(-1, 4)
            img_num = len(pred_phrases)
            bounding_boxes_per_image_per_phrase = {}
            all_object_names = []
            for i in range(img_num):
                for j, phrase in enumerate(pred_phrases[i].split(',')):
                    all_object_names.append(phrase)
            object_name_to_box_index = {}
            for i, object_name in enumerate(all_object_names):
                object_name_to_box_index[object_name] = i
            for i in range(img_num):
                bounding_boxes_per_image_per_phrase[i] = {}
                for j, phrase in enumerate(pred_phrases[i].split(',')):
                    bounding_boxes_per_image_per_phrase[i][phrase] = bounding_boxes[
                        object_name_to_box_index[phrase]]
            img_phrase_box_dict = {}
            for i in range(img_num):
                img_phrase_box_dict[i] = {}
                for j, phrase in enumerate(pred_phrases[i].split(',')):
                    map_coord = pixel_to_map_coord(pixel_x=int((bounding_boxes_per_image_per_phrase[i][phrase][0] +
                                                                bounding_boxes_per_image_per_phrase[i][phrase][2]) / 2),
                                                   pixel_y=int((bounding_boxes_per_image_per_phrase[i][phrase][1] +
                                                                bounding_boxes_per_image_per_phrase[i][phrase][3]) / 2),
                                                   depth_image=depth_images[i], camera_matrix=camera_matrix,
                                                   camera_pose=transforms[i])
                    # skip the object that contain nan coord
                    if np.isnan(map_coord).any():
                        continue
                    img_phrase_box_dict[i][phrase] = {}
                    img_phrase_box_dict[i][phrase]['bbox'] = bounding_boxes_per_image_per_phrase[i][phrase]
                    img_phrase_box_dict[i][phrase]['map_coord'] = map_coord
            # --- Filter Object
            # Distance >= 1.5, height >= 0.13
            distance_threshold = 1.5
            height_threshold = 0.13
            result_dict = {}
            current_transition, _ = self.state_machine.agent_controller.get_current_position()
            current_position = [current_transition.x, current_transition.y]
            for img_id in img_phrase_box_dict:
                result_dict[img_id] = {}
                for phrase in img_phrase_box_dict[img_id]:
                    map_coord = img_phrase_box_dict[img_id][phrase]['map_coord']
                    distance = np.linalg.norm(np.array(map_coord[:2]) - np.array(current_position))
                    height = map_coord[2]
                    if distance >= distance_threshold and height > height_threshold:
                        result_dict[img_id][phrase] = img_phrase_box_dict[img_id][phrase]
            # Filter object name is floor or ceiling
            ignore_object_name_list = ['floor', 'ceiling', 'wall']
            new_result_dict = {}
            for img_id in result_dict:
                new_result_dict[img_id] = {}
                for phrase in result_dict[img_id]:
                    for ignore_object_name in ignore_object_name_list:
                        if ignore_object_name not in phrase:
                            new_result_dict[img_id][phrase] = result_dict[img_id][phrase]
            result_dict = new_result_dict
            # 给找到的物品位置加上一定的噪声扰动，但偏差不超过原来的0.5m
            for img_id in result_dict:
                for phrase in result_dict[img_id]:
                    result_dict[img_id][phrase]['map_coord'][:3] += np.random.uniform(-0.5, 0.5, 3)

            self.state_machine.object_decision['bounding_boxes'] = bounding_boxes
            self.state_machine.object_decision['pred_phrases'] = pred_phrases
            self.state_machine.object_decision['result_dict'] = result_dict
            rospy.loginfo(f"ObjectDetection Results: {result_dict}")
            # if final result is empty
            if len(result_dict) == 0:
                rospy.logwarn("ObjectDetection: Failed, No valid object detected, Transition to NearestFrontier")
                return False
            return True
        except Exception as e:
            rospy.logerr(f"Exception in Surrounding ObjectDetection: {e}")
            return False


def visualize_object_in_rviz(map_coord, phrase, position_tolerance, marker_pub):
    rospy.logdebug("Starting visualization in rviz")
    marker_array = MarkerArray()
    maker_id = 0

    # Sphere marker
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = marker.SPHERE
    marker.id = maker_id
    maker_id += 1
    marker.action = marker.ADD
    marker.scale.x = 2 * position_tolerance
    marker.scale.y = 2 * position_tolerance
    marker.scale.z = 2 * position_tolerance
    marker.color.a = 0.4
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = map_coord[0]
    marker.pose.position.y = map_coord[1]
    marker.pose.position.z = map_coord[2]
    marker_array.markers.append(marker)

    # Text marker
    textmarker = Marker()
    textmarker.header.frame_id = "map"
    textmarker.type = textmarker.TEXT_VIEW_FACING
    textmarker.id = maker_id
    maker_id += 1
    textmarker.action = textmarker.ADD
    textmarker.scale.z = 1
    textmarker.color.a = 1.0
    textmarker.color.r = 1.0
    textmarker.color.g = 0.0
    textmarker.color.b = 0.0
    textmarker.pose.orientation.w = 1.0
    textmarker.pose.position.x = map_coord[0]
    textmarker.pose.position.y = map_coord[1]
    textmarker.pose.position.z = map_coord[2] + 0.3
    textmarker.text = phrase
    marker_array.markers.append(textmarker)

    # Publish the MarkerArray message
    marker_pub.publish(marker_array)
    rospy.logdebug("Finished visualization in rviz")


def ask_gpt_for_target_object_list(known_object_list, visited_object_list=[], final_object_target='tv_monitor',
                                   model="gpt-4") -> list:
    client = OpenAI()
    content_dict = {
        "target_object": final_object_target,
        "known_objects": known_object_list,
        "last_visited_objects": visited_object_list,
        # "ignore_object_name": ['floor', 'ceiling', 'wall']
    }
    content_str = json.dumps(content_dict)
    # prompt_messages_list = [
    #     {
    #         "role": "system",
    #         "content": "System: As an AI assistant, your role is to support a navigation agent operating within a residential setting. Your task involves analyzing a specific target object name and organizing a list of known objects that the navigation agent has already encountered and can access. Use commonsense reasoning to prioritize the objects in the 'known_objects' list based on the likelihood of the target object being found in proximity to these items. Additionally, consider the 'last_visited_objects' in your ranking, as these may influence the object's location. Exclude any objects from the 'known_objects' list that closely resemble those in the 'ignore_object_name' list (e.g., avoiding 'tile wall' when 'wall' is listed, or 'wooden floor' when 'floor' is mentioned). Ensure your output strictly adheres to the JSON format, consisting only of {\"ranked_objects\": []}. Avoid including any extraneous information in your response to eliminate processing errors.Here are two examples, followed by a query for your evaluation.\nUser: {\"target_object\":\"Find a tv\",\"known_objects\":[\"0_cabinet(0.45)\", \"0_stool(0.41)\", \"0_floor(0.39)\", \"0_room(0.34)\", \"0_cabinet(0.26)\", \"1_window(0.45)\", \"1_carpet(0.42)\", \"1_building(0.40)\", \"1_ceiling(0.31)\", \"2_ball(0.47)\", \"2_chair(0.45)\", \"2_stool(0.41)\", \"2_chair(0.38)\", \"2_room(0.32)\", \"2_floor(0.31)\", \"2_mat(0.28)\", \"3_table(0.44)\", \"3_chair(0.41)\", \"3_living room room(0.32)\", \"3_floor(0.26)\"],\"last_visited_objects\":[]}\nAI: {\"ranked_objects\": [\"3_living room room(0.32)\", \"0_cabinet(0.45)\", \"0_cabinet(0.26)\", \"3_table(0.44)\", \"2_chair(0.45)\", \"3_chair(0.41)\", \"2_chair(0.38)\", \"0_stool(0.41)\", \"2_stool(0.41)\", \"0_room(0.34)\", \"2_room(0.32)\", \"1_window(0.45)\", \"1_carpet(0.42)\", \"1_building(0.40)\", \"2_ball(0.47)\", \"0_floor(0.39)\", \"2_floor(0.31)\", \"3_floor(0.26)\", \"1_ceiling(0.31)\", \"2_mat(0.28)\"]}"
    #     },
    #     {
    #         "role": "user",
    #         "content": f"User: {content_str}"
    #     }
    # ]
    prompt_messages_list = [
        {
            "role": "system",
            "content": "System: As an AI assistant, your role is to support a navigation agent operating within a residential setting. Your task involves analyzing a specific target object name and organizing a list of known objects that the navigation agent has already encountered and can access. Use commonsense reasoning to prioritize the objects in the 'known_objects' list based on the likelihood of the target object being found in proximity to these items. Ensure your output strictly adheres to the JSON format, consisting only of {\\\"ranked_objects\\\": []}. Avoid including any extraneous information in your response to eliminate processing errors.Don't choose obejct whose name too general, like `floor`, `wall`, `building`, it is not helpful."
        },
        {
            "role": "user",
            "content": "{\"target_object\":\"Find a tv\",\"known_objects\":[\"0_cabinet(0.45)\", \"0_stool(0.41)\", \"0_floor(0.39)\", \"0_room(0.34)\", \"0_cabinet(0.26)\", \"1_window(0.45)\", \"1_carpet(0.42)\", \"1_building(0.40)\", \"1_ceiling(0.31)\", \"2_ball(0.47)\", \"2_chair(0.45)\", \"2_stool(0.41)\", \"2_chair(0.38)\", \"2_room(0.32)\", \"2_floor(0.31)\", \"2_mat(0.28)\", \"3_table(0.44)\", \"3_chair(0.41)\", \"3_living room room(0.32)\", \"3_floor(0.26)\"]}"
        },
        {
            "role": "assistant",
            "content": "{\"ranked_objects\": [\"0_cabinet(0.45)\", \"3_table(0.44)\", \"3_chair(0.41)\", \"0_stool(0.41)\", \"2_chair(0.45)\", \"2_stool(0.41)\", \"1_carpet(0.42)\", \"2_mat(0.28)\", \"3_living room room(0.32)\"]}"
        },
        {
            "role": "user",
            "content": content_str
        }
    ]
    # rospy.logdebug(f"Full GPT Prompt: {prompt_messages_list}")
    response = client.chat.completions.create(
        model=model,
        messages=prompt_messages_list,
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    ranked_object_list = json.loads(response.choices[-1].message.content)['ranked_objects']
    rospy.loginfo(f"GPT Response: {ranked_object_list}")
    return ranked_object_list


class GPTChooseObjectState(StateBase):
    def process_state(self):
        self.connection_failed = False
        self.novalid_object = False
        rospy.logdebug("GPT Choose Object Processing")
        success = self.perform_action()  # Placeholder for actual logic
        if success:
            rospy.logdebug("GPT Choose Object: Success, Transition to GotoPos")
            self.state_machine.try_set_state_by_name("goto_pos")
            return ProcessTextResponse("GPT choose object successful")
        else:
            if self.connection_failed:
                rospy.logwarn("GPT Choose Object: Failed due to connection error, Transition to GPTChooseObject")
                self.state_machine.status_pub.publish("Woops, I can't connect to GPT, I'll try again")
                self.state_machine.try_set_state_by_name("gpt_choose_object")
                return ProcessTextResponse("GPT choose object failed")
            if len(self.state_machine.object_decision['result_dict']) == 0:
                rospy.logwarn("GPT Choose Object: Failed, No Objects, Transition to NearestFrontier")
                self.state_machine.status_pub.publish("Woops, I can't find any object, I'll go to a nearest frontier instead")
                self.state_machine.try_set_state_by_name("nearest_frontier")
                return ProcessTextResponse("GPT choose object failed")
            if self.novalid_object:
                rospy.logwarn("GPT Choose Object: Failed, No valid object detected, Transition to NearestFrontier")
                self.state_machine.status_pub.publish("Woops, I can't find any valid object, I'll go to a nearest frontier instead")
                self.state_machine.try_set_state_by_name("nearest_frontier")
                return ProcessTextResponse("GPT choose object failed")
            rospy.logwarn("GPT Choose Object: Failed, Transition to NearestFrontier")
            self.state_machine.try_set_state_by_name("nearest_frontier")
            self.state_machine.status_pub.publish(
                "Woops, Something wrong, I'll go to a nearest frontier instead")
            return ProcessTextResponse("GPT choose object failed")

    def perform_action(self):
        self.state_machine.status_pub.publish("I'm asking GPT to choose an object for next target...")
        rospy.logdebug("GPT Choose Object Performing")
        try:
            result_dict = self.state_machine.object_decision['result_dict']
            imgid_phrase_coord_dict = {}
            for img_id in result_dict:
                for phrase in result_dict[img_id]:
                    imgid_phrase_coord_dict[f"{img_id}_{phrase}"] = {}
                    imgid_phrase_coord_dict[f"{img_id}_{phrase}"]['map_coord'] = result_dict[img_id][phrase]['map_coord']
                    imgid_phrase_coord_dict[f"{img_id}_{phrase}"]['img_id'] = img_id
                    imgid_phrase_coord_dict[f"{img_id}_{phrase}"]['phrase'] = phrase
            all_id_object_names = list(imgid_phrase_coord_dict.keys())

            sorted_object_name_list = ask_gpt_for_target_object_list(known_object_list=all_id_object_names,
                                                                     visited_object_list=self.state_machine.visited_object_list,
                                                                     final_object_target=self.state_machine.target)
            if len(sorted_object_name_list) == 0:
                rospy.logwarn("GPT Choose Object: Failed, No valid object detected, Transition to NearestFrontier")
                self.novalid_object = True
                return False
            # Get the top ranked object
            chosen_object_name = sorted_object_name_list[0]
            img_id, phrase = imgid_phrase_coord_dict[chosen_object_name]['img_id'], \
                imgid_phrase_coord_dict[chosen_object_name]['phrase']
            img_id = int(img_id)
            rospy.loginfo(f"GPT Choose Object: {phrase}")
            self.state_machine.status_pub.publish(f"GPT Choose Object: {phrase}")
            self.state_machine.object_decision['chosen'] = {"img_id": img_id, "phrase": phrase}
            # Get the map coordinate of the selected object
            map_coord = result_dict[img_id][phrase]['map_coord']
            rospy.loginfo(f"GPT Choose Object Map Coord: {map_coord}")
            rospy.logdebug("GPT Choose Object: Start Publishing MarkerArray for visualization in rviz")
            # Visualize the selected object in rviz
            visualize_object_in_rviz(map_coord=result_dict[img_id][phrase]['map_coord'], phrase=phrase,
                                     position_tolerance=self.state_machine.position_tolerance,
                                     marker_pub=self.state_machine.agent_controller.marker_pub)
            rospy.logdebug("Random Choose Object: Finished Publishing MarkerArray for visualization in rviz")
            return True
        except APIConnectionError as e:
            rospy.logerr(f"APIConnectionError in GPT Choose Object: {e}")
            self.connection_failed = True
            return False
        except Exception as e:
            rospy.logerr(f"Exception in GPT Choose Object: {e}")
            return False


class RandomChooseObjectState(StateBase):
    def process_state(self):
        rospy.logdebug("Random Choose Object Processing")
        success = self.perform_action()  # Placeholder for actual logic
        if success:
            rospy.logdebug("Random Choose Object: Success, Transition to GotoPos")
            self.state_machine.try_set_state_by_name("goto_pos")
            return ProcessTextResponse("Random choose object processed")
        else:
            if len(self.state_machine.object_decision['result_dict']) == 0:
                rospy.logwarn("Random Choose Object: Failed, No Objects, Transition to Idle")
                self.state_machine.try_set_state_by_name("idle")
                return ProcessTextResponse("Random choose object failed")
            rospy.logwarn("Random Choose Object: Failed, Transition to RandomChooseObject")
            self.state_machine.try_set_state_by_name("random_choose_object")
            return ProcessTextResponse("Random choose object failed")

    def perform_action(self):
        self.state_machine.status_pub.publish(f"I'll randomly choose an object for next target...")
        rospy.logdebug("Random Choose Object Performing")
        try:
            result_dict = self.state_machine.object_decision['result_dict']
            # filter key with empty value
            result_dict = {k: v for k, v in result_dict.items() if v}
            # Random Choose one object
            img_id = random.choice(list(result_dict.keys()))
            phrase = random.choice(list(result_dict[img_id].keys()))
            rospy.loginfo(f"Random Choose Object: {phrase}")
            rospy.loginfo(f"Random Choose Object: {phrase}")
            self.state_machine.object_decision['chosen'] = {"img_id": img_id, "phrase": phrase}
            # Get the map coordinate of the selected object
            map_coord = result_dict[img_id][phrase]['map_coord']
            rospy.loginfo(f"Random Choose Object Map Coord: {map_coord}")
            rospy.logdebug("Random Choose Object: Start Publishing MarkerArray for visualization in rviz")
            self.state_machine.status_pub.publish(f"Random Choose Object: {phrase}")
            # Visualize the selected object in rviz
            visualize_object_in_rviz(map_coord=map_coord, phrase=phrase,
                                     position_tolerance=self.state_machine.position_tolerance,
                                     marker_pub=self.state_machine.agent_controller.marker_pub)

            rospy.logdebug("Random Choose Object: Finished Publishing MarkerArray for visualization in rviz")
            # ------ Visualize end ------
            return True
        except Exception as e:
            rospy.logerr(f"Exception in Random Choose Object: {e}")
            return False


class RandomChooseFrontierState(StateBase):
    def process_state(self):
        rospy.logdebug("Random Choose Frontier Processing")
        success = self.perform_action()  # Placeholder for actual logic
        if success:
            rospy.logdebug("Random Choose Frontier: Success, Transition to GotoPos")
            self.state_machine.try_set_state_by_name("goto_pos")
            return ProcessTextResponse("Random choose frontier processed")
        else:
            if len(self.state_machine.agent_controller.frontiers) == 0:
                rospy.logwarn("Random Choose Frontier: Failed, No Frontiers, Transition to Idle")
                self.state_machine.try_set_state_by_name("idle")
                return ProcessTextResponse("Random choose frontier failed")
            rospy.logwarn("Random Choose Frontier: Failed, Transition to RandomChooseFrontier")
            self.state_machine.try_set_state_by_name("random_choose_frontier")
            return ProcessTextResponse("Random choose frontier failed")

    def perform_action(self):
        self.state_machine.status_pub.publish(f"I'll randomly choose a frontier for next target...")
        rospy.logdebug("Random Choose Frontier Performing")
        try:
            img_id = 0
            phrase = "chosen_frontier"
            frontiers_points = self.state_machine.agent_controller.get_frontier()
            # Random Choose one object
            point = random.choice(frontiers_points)
            rospy.loginfo(f"Random Choose Frontier: {point}")
            self.state_machine.object_decision['chosen'] = {}
            self.state_machine.object_decision['chosen']['point'] = point
            self.state_machine.object_decision['chosen']['phrase'] = "chosen_frontier"
            self.state_machine.object_decision['chosen']['img_id'] = 0
            # Visualize the selected point in rviz
            visualize_object_in_rviz(map_coord=point, phrase="chosen_frontier",
                                     position_tolerance=self.state_machine.position_tolerance,
                                     marker_pub=self.state_machine.agent_controller.marker_pub)
            return True
        except Exception as e:
            rospy.logerr(f"Exception in Random Choose Object: {e}")
            return False

class ChooseNearestFrontierState(StateBase):
    def process_state(self):
        rospy.logdebug("Choose Nearest Frontier Processing")
        success = self.perform_action()  # Placeholder for actual logic
        if success:
            rospy.logdebug("Choose Nearest Frontier: Success, Transition to GotoPos")
            self.state_machine.try_set_state_by_name("goto_pos")
            return ProcessTextResponse("Choose Nearest Frontier processed")
        else:
            if len(self.state_machine.agent_controller.frontiers) == 0:
                rospy.logwarn("Choose Nearest Frontier: Failed, No Frontiers, Transition to Idle")
                self.state_machine.try_set_state_by_name("idle")
                return ProcessTextResponse("Choose Nearest Frontier failed")
            rospy.logwarn("Choose Nearest Frontier: Failed, Transition to RandomChooseFrontier")
            self.state_machine.try_set_state_by_name("random_choose_frontier")
            return ProcessTextResponse("Choose Nearest Frontier failed")

    def perform_action(self):
        rospy.logdebug("Choose Nearest Frontier Performing")
        self.state_machine.status_pub.publish(f"I'll choose the nearest frontier for next target...")
        try:
            frontiers_points = self.state_machine.agent_controller.get_frontier()
            # Choose the nearest frontier
            current_transition, _ = self.state_machine.agent_controller.get_current_position()
            current_position = [current_transition.x, current_transition.y]
            current_position = np.array(current_position)[:2]
            frontiers_points_position = np.array(frontiers_points)[:,:2]
            distance = np.linalg.norm(frontiers_points_position - current_position, axis=1)
            # distance < 1.5m 的不考虑
            distance[distance < 1.5] = 1000
            min_index = np.argmin(distance)
            point = frontiers_points[min_index]
            rospy.loginfo(f"Choose Nearest Frontier: {point}")
            self.state_machine.object_decision['chosen'] = {}
            self.state_machine.object_decision['chosen']['point'] = point
            self.state_machine.object_decision['chosen']['phrase'] = "chosen_frontier"
            self.state_machine.object_decision['chosen']['img_id'] = 0
            # Visualize the selected point in rviz
            visualize_object_in_rviz(map_coord=point, phrase="chosen_frontier",
                                     position_tolerance=self.state_machine.position_tolerance,
                                     marker_pub=self.state_machine.agent_controller.marker_pub)
            return True
        except Exception as e:
            rospy.logerr(f"Exception in Choose Nearest Frontier: {e}")
            return False

class GotoPosState(StateBase):
    def process_state(self):
        rospy.logdebug("Go to position Processing")
        self.state_machine.status_pub.publish(f"Going to the position...")
        success = self.perform_action()  # Placeholder for actual logic
        if success:
            rospy.logdebug("Go to position: Position Sent, Transition to PositionMonitor")
            self.state_machine.try_set_state_by_name("pos_monitor")
            return ProcessTextResponse("Go to position: Send Postion to move_base Successful")
        else:
            rospy.logwarn("Go to position: Send Position Failed, Transition to LookSurrounding")
            self.state_machine.try_set_state_by_name("look_surrounding")
            return ProcessTextResponse("Go to position: Send Postion to move_base Failed")

    def perform_action(self):
        try:
            if "point" in self.state_machine.object_decision['chosen']:
                point = self.state_machine.object_decision['chosen']['point']
                rospy.logdebug(f"Go to position Performing: Got Selected Frontier: {point}")
                rospy.logdebug("Go to position Performing: Sending Position to move_base")
                self.state_machine.agent_controller.nav2point(point[0], point[1], wait_for_result=False)
                rospy.loginfo("Go to position Performing: Position Sent")
            else:
                result_dict = self.state_machine.object_decision['result_dict']
                img_id, phrase = self.state_machine.object_decision['chosen']['img_id'], \
                    self.state_machine.object_decision['chosen']['phrase']
                map_coord = result_dict[img_id][phrase]['map_coord']
                rospy.logdebug(f"Go to position Performing: Got Selected Object: Name: {phrase}, Map Coord: {map_coord}")
                rospy.logdebug("Go to position Performing: Sending Position to move_base")
                self.state_machine.agent_controller.nav2point(map_coord[0], map_coord[1], wait_for_result=False)
                # self.state_machine.agent_controller.client
                rospy.loginfo("Go to position Performing: Position Sent")
            return True
        except Exception as e:
            rospy.logerr(f"Exception in Go to position Performing: {e}")
            return False


class PositionMonitorState(StateBase):
    def __init__(self, id, name, detail, state_machine, duration=50):
        super(PositionMonitorState, self).__init__(id, name, detail, state_machine)
        self.start_time = rospy.get_time()
        self.duration = duration
        self.stuck_count_tolerance = 20

    def reset_timer(self):
        self.start_time = rospy.get_time()
        self.stuck_count = 0
        self.last_position = None

    def process_state(self):
        rospy.logdebug("Position Monitor Processing")
        if self.state_machine.last_state.name != "pos_monitor":
            self.state_machine.status_pub.publish(f"Walking to the position...")
            self.reset_timer()
        success = self.perform_action()
        if success:
            self.state_machine.status_pub.publish(f"I'm Arrived!")
            rospy.logdebug("Position Monitor: Success, Stop Moving, Cancel move_base goal")
            self.state_machine.object_decision['chosen'].pop('point', None)
            rospy.logdebug("Position Monitor: Success, Transition to WhetherFinished")
            self.state_machine.try_set_state_by_name("whether_finsihed")
            return ProcessTextResponse("Position Monitor Result: Go to position successful")
        else:
            # if self.stuck_count >= self.stuck_count_tolerance:
            #     if self.state_machine.object_decision['chosen'].get('phrase', None) == "chosen_frontier":
            #         self.state_machine.status_pub.publish(f"Something wrong, I'm Stucked on my way to nearset frontier, I'll randomly choose a frontier for next target...")
            #         rospy.logerr("Position Monitor: Stucked, Transition to RandomChooseFrontier")
            #         self.state_machine.try_set_state_by_name("random_choose_frontier")
            #         return ProcessTextResponse("Position Monitor Result: Stucked")
            #     else:
            #         self.state_machine.status_pub.publish(f"Something wrong, I'm Stucked on my way to the chosen object, I'll choose the nearest frontier for next target...")
            #         rospy.logerr("Position Monitor: Stucked, Transition to NearestFrontier")
            #         self.state_machine.try_set_state_by_name("nearest_frontier")
            #         return ProcessTextResponse("Position Monitor Result: Stucked")

            if rospy.get_time() - self.start_time > self.duration:
                if self.state_machine.object_decision['chosen'].get('phrase', None) == "chosen_frontier":
                    self.state_machine.status_pub.publish(f"It takes too long to go to the nearest frontier, I'll randomly choose a frontier for next target...")
                    rospy.logwarn("Position Monitor: Failed to nearest frontier due to timeout, Transition to RandomChooseFrontier")
                    self.state_machine.try_set_state_by_name("random_choose_frontier")
                    return ProcessTextResponse("Position Monitor Status: Failed to nearest frontier due to timeout")
                else:
                    self.state_machine.status_pub.publish(f"It takes too long to go to the chosen object, I'll choose the nearest frontier for next target...")
                    rospy.logwarn("Position Monitor: Failed due to timeout, Transition to NearestFrontier")
                    self.state_machine.try_set_state_by_name("nearest_frontier")
                    return ProcessTextResponse("Position Monitor Status: Failed due to timeout")
            else:
                rospy.logwarn("Position Monitor: Not arrived yet, Continue to Monitor")
                self.state_machine.try_set_state_by_name("pos_monitor")
            return ProcessTextResponse("Position Monitor Status: Continue to Monitor")

    def perform_action(self):
        try:
            rospy.logdebug("Position Monitor Performing")
            # Target Position
            if "point" in self.state_machine.object_decision['chosen']:
                img_id, phrase = 0, "chosen_frontier"
                target_map_coord = self.state_machine.object_decision['chosen']['point'][:2]
                # position_tolerance = 0.5
            else:
                result_dict = self.state_machine.object_decision['result_dict']
                img_id, phrase = self.state_machine.object_decision['chosen']['img_id'], \
                    self.state_machine.object_decision['chosen']['phrase']
                target_map_coord = result_dict[img_id][phrase]['map_coord'][:2]
                # position_tolerance = self.state_machine.position_tolerance
            # Get Current PositionA
            current_transition, _ = self.state_machine.agent_controller.get_current_position()
            current_position = [current_transition.x, current_transition.y]

            # Calculate Distance
            distance = np.linalg.norm(np.array(target_map_coord) - np.array(current_position))


            # If last position and current position are very close(say <=0.1m), then stack_count + 1
            if self.last_position is not None:
                if np.linalg.norm(np.array(self.last_position) - np.array(current_position)) <= 0.01:
                    self.stuck_count += 1
                else:
                    self.stuck_count = 0

            # if self.stuck_count >= self.stuck_count_tolerance:
            #     rospy.logerr("Position Monitor: Stucked, Cancel move_base goal")
            #     self.state_machine.agent_controller.client.cancel_goal()
            #     return False
            #
            self.last_position = current_position

            rospy.loginfo(f"""
            Position Monitor Performing:
            Target Position: {target_map_coord}
            Current Position: {current_position}
            Distance: {distance}
            Tolernace: {self.state_machine.position_tolerance}m
            """)

            if rospy.get_time() - self.start_time > self.duration:
                rospy.logerr("Position Monitor: Timeout")
                self.state_machine.agent_controller.client.cancel_goal()
                return False

            # Check if distance is within tolerance
            if distance < self.state_machine.position_tolerance:
                rospy.loginfo("Position Monitor: Success, Cancel move_base goal")
                self.state_machine.agent_controller.client.cancel_goal()
                self.state_machine.visited_object_list.append(f"{img_id}_{phrase}")
                return True
            else:
                return False
        except Exception as e:
            rospy.logerr(f"Exception in Position Monitor:{e}")
            self.state_machine.agent_controller.client.cancel_goal()
            return False


def ask_gpt_whether_met_goal(object_name, target):
    client = OpenAI()
    query_dict = {"object_name": f"{object_name}", "target": f"{target}"}
    query_dict_str = json.dumps(query_dict)
    messages = [
        {
            "role": "system",
            "content": "System: You are a sophisticated navigation AI with a specialized function. Your task is to determine if the currently identified object, denoted by its name, aligns with a given target. It's important to recognize that the object name and the target need not match exactly; a close similarity is adequate. Crucially, your responses must be formatted in JSON, either as {\"result\": \"True\"} or {\"result\": \"False\"}. No other content should be included in your response to avoid errors in further processing. Here are two examples, followed by a query for your evaluation.\nUser: {\"object_name\":\"sofa(0.25)\",\"target\":\"Find a chair\"}\nAI: {\"result\":\"False\"}\nUser: {\"object_name\":\"tv_monitor(0.35)\",\"target\":\"Find a chair\"}\nAI: {\"result\":\"False\"}"
        },
        {
            "role": "user",
            "content": query_dict_str
        }
    ]
    messages_str = json.dumps(messages)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    result = response.choices[0].message.content
    result_dict = json.loads(result)
    rospy.logdebug(f"Full GPT Prompt: {messages_str}")
    rospy.logdebug(f"GPT Response: {result}")
    return result_dict['result'] == 'True'


class WhetherFinishedState(StateBase):
    def process_state(self):
        self.connection_failed = False
        rospy.logdebug("Whether finished Processing")
        finished = self.perform_action()  # Placeholder for actual logic
        if finished:
            rospy.logdebug("Whether finished: Yes, Transition to Idle")
            self.state_machine.status_pub.publish(f"Yep, I've met the goal{self.state_machine.target}!")
            self.state_machine.visited_object_list = []
            self.state_machine.try_set_state_by_name("idle")
            return ProcessTextResponse("Task finished")
        else:
            if self.connection_failed:
                rospy.logwarn("Whether finished: Failed due to connection error, Transition to WhetherFinished")
                self.state_machine.status_pub.publish(f"Woops, I can't connect to GPT, I'll try again...")
                self.state_machine.try_set_state_by_name("whether_finsihed")
            rospy.logdebug("Whether finished: No, Transition to LookSurrounding")
            self.state_machine.status_pub.publish(f"Nope, I haven't met the goal yet, I'll look around again...")
            self.state_machine.try_set_state_by_name("look_surrounding")
            return ProcessTextResponse("Task not finished, restarting")

    def perform_action(self):
        rospy.logdebug("Whether finished Performing")
        try:
            self.state_machine.status_pub.publish(f"I've arrived at {self.state_machine.object_decision['chosen']['phrase']},\n"
                                                  f"I'm asking GPT whether that means I've met the goal{self.state_machine.target}...")
            result = ask_gpt_whether_met_goal(object_name=self.state_machine.object_decision['chosen']['phrase'],
                                              target=self.state_machine.target)
            rospy.loginfo(f"Whether finished Performing: Result: {result}")
            return result
        except APIConnectionError as e:
            rospy.logerr(f"APIConnectionError in Whether finished Performing: {e}")
            self.connection_failed = True
            return False
        except Exception as e:
            rospy.logerr(f"Exception in Whether finished Performing: {e}")
            return False


class StateMachine(object):
    def __init__(self):
        self.lock = threading.Lock()  # 添加一个线程锁
        self.states = {
            "idle": StateBase(id=0, name="idle", detail="Idle state", state_machine=self),
            "look_surrounding": LookSurroundingState(id=1, name="look_surrounding", detail="Looking surrounding",
                                                     state_machine=self),
            "object_detect": ObjectDetectionState(id=2, name="object_detect", detail="Detecting surrounding objects",
                                                  state_machine=self),
            "gpt_choose_object": GPTChooseObjectState(id=3, name="gpt_choose_object",
                                                      detail="Using GPT to decide which object to pick",
                                                      state_machine=self),
            "goto_pos": GotoPosState(id=4, name="goto_pos", detail="Going to position", state_machine=self),
            "pos_monitor": PositionMonitorState(id=5, name="pos_monitor", detail="Monitoring position",
                                                state_machine=self),
            "random_choose_frontier": RandomChooseFrontierState(id=6, name="random_choose_frontier",
                                                                    detail="Randomly choose frontier",
                                                                    state_machine=self),
            "nearest_frontier": ChooseNearestFrontierState(id=7, name="nearest_frontier",
                                                                detail="Choose nearest frontier",
                                                                state_machine=self),
            "random_choose_object": RandomChooseObjectState(id=7, name="random_choose_object",
                                                            detail="Randomly choose object", state_machine=self),
            "whether_finsihed": WhetherFinishedState(id=8, name="whether_finsihed",
                                                     detail="Checking if the task is finished", state_machine=self),
        }
        self.last_state = None
        self.state = self.states["idle"]
        self.state_id_to_name = {state.id: state.name for state in self.states.values()}
        self.state_name_to_id = {state.name: state.id for state in self.states.values()}
        self.transition_rule_table = {
            "idle": ["look_surrounding"],
            "look_surrounding": ["object_detect", "idle"],
            "object_detect": ["gpt_choose_object", "idle","nearest_frontier"],
            "gpt_choose_object": ["goto_pos","gpt_choose_object","random_choose_object", "nearest_frontier"],
            "goto_pos": ["pos_monitor", "random_choose_object"],
            "pos_monitor": ["whether_finsihed", "nearest_frontier", "pos_monitor", "look_surrounding", "random_choose_frontier"],
            "random_choose_object": ["goto_pos", "random_choose_object", "idle"],
            "random_choose_frontier": ["goto_pos", "random_choose_frontier", "idle"],
            "nearest_frontier": ["goto_pos", "random_choose_frontier", "idle"],
            "whether_finsihed": ["idle", "look_surrounding", "whether_finsihed"],
        }
        self.bridge = CvBridge()
        self.agent_controller = ROSAgentController(init_node=False)
        rospy.loginfo("Waiting for Multiple ImageProcessing service...")
        rospy.wait_for_service('image_processing/multiple')
        rospy.loginfo("Multiple ImageProcessing service is available.")
        self.multi_image_processing_service_proxy = rospy.ServiceProxy('image_processing/multiple',
                                                                       MultiImageProcessing)
        rospy.loginfo("Connected to Multiple ImageProcessing service.")

        self.object_decision = {}
        self.position_tolerance = 1.5
        self.target = None
        self.visited_object_list = []

        self.status_pub = rospy.Publisher('current_status', String, queue_size=10)

    def process_current_state(self):
        current_state = self.get_state().name
        if current_state == "idle":
            return
        elif current_state in self.states:
            self.states[current_state].process_state()
        else:
            rospy.logerr(f"Unknown state {current_state}")

    def can_transition_to(self, new_state):
        return new_state in self.transition_rule_table[self.state.name]

    def try_set_state_by_name(self, new_state_name):
        with self.lock:  # 使用锁来保护状态更改
            if new_state_name in self.states and self.can_transition_to(new_state_name):
                if new_state_name == "idle":
                    self.agent_controller.client.cancel_goal()
                    self.visited_object_list = []
                self.last_state = self.state
                self.state = self.states[new_state_name]
                # if state has set method, call it
                if hasattr(self.state, 'select'):
                    self.state.select()
            else:
                raise Exception(f"Invalid or disallowed state transition to {new_state_name}")

    def try_set_state_by_id(self, state_id):
        state_name = self.state_id_to_name[state_id]
        self.try_set_state_by_name(state_name)

    def get_state(self):
        return self.state


class IDLEServer(object):
    def __init__(self, log_level=rospy.INFO):
        rospy.init_node('idle_interface_server', log_level=log_level)
        self.state_machine = StateMachine()
        self.service = rospy.Service('process_text', ProcessText, self.handle_process_text)
        rospy.loginfo("[IDLEServer]: Ready to recieve text.")
        self.state_machine.status_pub.publish("Ready")
        self.get_state_service = rospy.Service('get_state', ProcessText, self.handle_get_state)
        self.openai_api_key = os.environ['OPENAI_API_KEY']

    def handle_process_text(self, req):
        rospy.logdebug(f"Received text: {req.input}")
        self.state_machine.status_pub.publish(f"Received Task: {req.input}")
        current_state = self.state_machine.get_state()
        if current_state.name != "idle":
            return ProcessTextResponse(f"Currently busy with: {current_state.detail}")
        self.start_process(req.input)
        return ProcessTextResponse("Starting task")

    def handle_get_state(self, req):
        current_state = self.state_machine.get_state()
        return GetStateResponse(current_state.id, current_state.name, current_state.detail)

    def start_process(self, input_text):
        if input_text:
            self.state_machine.target = input_text
            self.state_machine.try_set_state_by_name("look_surrounding")
        else:
            rospy.loginfo("No input text received, staying in idle state")

    def main_loop(self, rate=10):
        rate = rospy.Rate(rate)  # 设置循环频率为10Hz
        while not rospy.is_shutdown():
            try:
                rospy.logdebug("Current state: %s", self.state_machine.get_state().name)
                self.state_machine.process_current_state()
                rate.sleep()  # 按照设置的频率暂停
            except rospy.ROSInterruptException:
                rospy.loginfo("IDLEServer node terminated.")
                break
            except Exception as e:
                rospy.logerr(f"Exception in main loop: {e}")


if __name__ == "__main__":
    try:
        server = IDLEServer(log_level=rospy.DEBUG)
        server.main_loop(rate=10)
    except rospy.ROSInterruptException:
        rospy.loginfo("IDLEServer node terminated.")
