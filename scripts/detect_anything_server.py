#!/usr/bin/env python
import os
import cv2
import torch
import rospy
from sensor_msgs.msg import Image
from PIL import Image as ImagePIL
from create_embeeding_grid_map.srv import ImageProcessing, ImageProcessingResponse
from create_embeeding_grid_map.srv import MultiImageProcessing, MultiImageProcessingResponse
from ram_plus.ram_dino import get_grounding_output, load_model, load_grounding_image, show_box
from cv_bridge import CvBridge
import numpy as np
import torchvision
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from ram.models import ram_plus
import matplotlib.pyplot as plt

def convert_to_rgb(image):
    return image.convert("RGB")

def inference_ram(image, model):
    with torch.no_grad():
        tags, tags_chinese = model.generate_tag(image)
    return tags[0],tags_chinese[0]



class RAMModelServer:
    def __init__(self, image_size=384, publish_bbox_image=False, ram_plus_checkpoint="ram_plus_swin_large_14m.pth", grounded_checkpoint = "groundingdino_swint_ogc.pth", config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", device="cuda") -> None:
        self.status = "standby"
        rospy.init_node('image_processing_server')
        self.device = device
        self.service = rospy.Service('image_processing/single', ImageProcessing, self.handle_image_processing)
        self.multi_image_service = rospy.Service('image_processing/multiple', MultiImageProcessing, self.handle_multi_image_processing)
        self.transform = Compose([
                                    convert_to_rgb,
                                    Resize((image_size, image_size)),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
        rospy.loginfo("Loading RAM model...")
        self.ram_model = ram_plus(pretrained=ram_plus_checkpoint, 
                                    image_size=image_size,
                                    vit='swin_l')
        self.ram_model.eval()
        self.ram_model = self.ram_model.to(self.device)
        self.box_threshold = 0.25
        self.text_threshold = 0.2
        self.iou_threshold = 0.5
        self.openai_key = os.environ["OPENAI_API_KEY"]
        rospy.loginfo("RAM model loaded.")
        rospy.loginfo("Loading GroundingDino model...")
        self.grounding_dino_model = load_model(config_file, grounded_checkpoint, device=device)
        rospy.loginfo("GroundingDino model loaded.")
        self.publish_bbox_image = publish_bbox_image
        if self.publish_bbox_image:
            self.pub_bbox_image = rospy.Publisher('bbox_image', Image, queue_size=10)
        self.status = "ready"
        rospy.loginfo("Ready to process images.")


    def process_single_image(self, image_msg, target=None):
        try:
            # Convert ROS Image message to OpenCV image
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Using RAM model to generate tags
            rospy.loginfo("RAM model is processing image...")
            image_pil = ImagePIL.fromarray(cv_image)
            grouding_image = load_grounding_image(image_pil)
            raw_image = image_pil.resize((384, 384))
            raw_image = self.transform(raw_image).unsqueeze(0).to(self.device)
            res = inference_ram(raw_image, self.ram_model)

            # Process tags
            tags = res[0].replace(' |', ',')
            tags_chinese = res[1].replace(' |', ',')
            # filted_tags_list = self.gpt_filter_out_not_object(tags)
            # filted_tags_list.append('ceiling', 'floor')
            # filted_tags_list = list(set(filted_tags_list))
            # tags = ','.join(filted_tags_list)

            # 将目标加入到tags中
            if target is not None or target != '':
                tags_list = tags.split(',')
                target = target.replace(' ', '_')
                if target not in tags_list:
                    tags_list.append(target)
                tags = ','.join(tags_list)

            rospy.loginfo("RAM model finished processing image. Result:")
            rospy.loginfo("Tags: %s" % tags)

            # Using GroundingDino Model to generate bounding boxes
            rospy.loginfo("GroundingDino model is processing image...")
            boxes_filt, scores, pred_phrases = get_grounding_output(
                self.grounding_dino_model, grouding_image, tags, self.box_threshold, self.text_threshold, device=self.device
            )
            rospy.loginfo(f"Pred phrases: {pred_phrases}")
            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]
            boxes_filt = boxes_filt.cpu()
            rospy.loginfo("GroundingDino model finished processing image.")

            # NMS
            rospy.loginfo(f"Before NMS: {boxes_filt.shape[0]} boxes")
            nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]
            pred_phrases = ','.join(pred_phrases)
            rospy.loginfo(f"After NMS: {boxes_filt.shape[0]} boxes")

            if self.publish_bbox_image:
                self.draw_and_publish_bbox_image(cv_image, boxes_filt, tags, tags_chinese, pred_phrases)

            return tags, tags_chinese, boxes_filt, pred_phrases
        except Exception as e:
            rospy.logerr("Image processing failed: %s" % e)
            return "", "", [0, 0, 0, 0], ""  # Assuming 4 values per box

    def handle_multi_image_processing(self, req):
        if self.status != "ready":
            rospy.logerr("Server is not ready to process images.")
            return MultiImageProcessingResponse([], [], [])
        rospy.loginfo("Received images, processing...")
        all_tags = []
        all_tags_chinese = []
        all_boxes = []
        all_pred_phrases = []
        for image_msg in req.images:
            tags, tags_chinese, boxes_filt, pred_phrases = self.process_single_image(image_msg, target=req.target)
            all_tags.append(tags)
            all_tags_chinese.append(tags_chinese)
            all_boxes.extend(boxes_filt)
            all_pred_phrases.append(pred_phrases)

        all_boxes = np.ravel(all_boxes).tolist()
        rospy.loginfo(f"All Pred Phrases: {all_pred_phrases}")
        return MultiImageProcessingResponse(all_pred_phrases, all_boxes)

    def handle_image_processing(self, req):
        if self.status != "ready":
            rospy.logerr("Server is not ready to process images.")
            return MultiImageProcessingResponse([], [], [])
        rospy.loginfo("Received an image, processing...")
        tags, tags_chinese, boxes_filt, pred_phrases = self.process_single_image(req.image, target=req.target)
        return ImageProcessingResponse(pred_phrases, np.ravel(boxes_filt).tolist())

    def draw_and_publish_bbox_image(self, cv_image, boxes_filt, tags, tags_chinese, pred_phrases):
        rospy.loginfo("Drawing and publishing image with bounding boxes...")
        # Convert OpenCV image to PIL image
        image_pil = ImagePIL.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Create a figure for drawing
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_pil)

        # Draw bounding boxes and labels
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box, ax, label)

        # Remove axes and title
        ax.set_axis_off()
        plt.title('RAM-tags: ' + tags)

        # Convert the matplotlib plot to OpenCV image
        fig.canvas.draw()
        bbox_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        bbox_image = bbox_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR)
        plt.close(fig)

        # Convert OpenCV image to ROS Image message
        bridge = CvBridge()
        ros_image = bridge.cv2_to_imgmsg(bbox_image, "bgr8")
    
        # Publish the image
        self.pub_bbox_image.publish(ros_image)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ram_plus_checkpoint= "/home/yinzi/catkin_ws_conda/src/create_embeeding_grid_map/scripts/ram_plus/model_checkpoints/ram_plus_swin_large_14m.pth"
    grounded_checkpoint = "/home/yinzi/catkin_ws_conda/src/create_embeeding_grid_map/scripts/ram_plus/model_checkpoints/groundingdino_swint_ogc.pth"
    config_file = "/home/yinzi/catkin_ws_conda/src/create_embeeding_grid_map/scripts/ram_plus/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    publish_bbox_image = False
    image_process_server = RAMModelServer(image_size=384, publish_bbox_image=publish_bbox_image,
                                            ram_plus_checkpoint=ram_plus_checkpoint, 
                                            grounded_checkpoint=grounded_checkpoint, 
                                            config_file=config_file, 
                                            device=device)
    rospy.spin()
