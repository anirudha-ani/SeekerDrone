import time
import numpy as np
import airsim
import os
import config
import cv2

clockspeed = 1
timeslice = 0.5 / clockspeed
goalY = 57
outY = -0.5
floorZ = 1.18
goals = [7, 17, 27.5, 45, goalY]
speed_limit = 0.2

class Env:
    def __init__(self):
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        with open("yolov3.txt", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

        self.goal = config.goal
        self.init_pos = self.client.getMultirotorState().kinematics_estimated.position

    def reset(self):
        self.level = 0
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # my takeoff
        self.client.simPause(False)
        self.client.moveByVelocityAsync(0, 0, -1, 2 * timeslice).join()
        self.client.moveByVelocityAsync(0, 0, 0, 0.1 * timeslice).join()
        self.client.hoverAsync().join()
        self.client.simPause(True)
        # quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        # responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        # quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        # observation = [responses, quad_vel]
        observation, Width, Height = self.detect_image()

        return observation

    def take_action(self, action):
        if action == 0:#w
            self.client.moveByVelocityBodyFrameAsync(1, 0, 0, 1).join()
        elif action == 1:#a
            self.client.rotateByYawRateAsync(-20,1).join()
        elif action == 2:#s
            self.client.moveByVelocityBodyFrameAsync(-1, 0, 0, 1).join()
        elif action == 3:#d
            self.client.rotateByYawRateAsync(20,1).join()
        elif action == 4:#u
            self.client.moveByVelocityBodyFrameAsync(0, 0, -1, 1).join()
        elif action == 5:#j
            self.client.moveByVelocityBodyFrameAsync(0, 0, 1, 1).join()

    def step(self, action):
        # move with given velocity
        self.client.simPause(False)

        has_collided = False
        self.take_action(action)
        start_time = time.time()
        while time.time() - start_time < timeslice:
            # get quadrotor states
            quad_pos = self.client.getMultirotorState().kinematics_estimated.position
            quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

            # decide whether collision occured
            collided = self.client.simGetCollisionInfo().has_collided

            if collided:
                has_collided = True
                break
        self.client.simPause(True)

        # observe with depth camera
        # responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        observation, Width, Height = self.detect_image()
        detected, num_goal = self.detected_obj(observation, Width, Height)

        # get quadrotor states
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position

        # dead: too far from original point or collided #todo
        dead = has_collided or (sum(map(lambda i : i * i, quad_pos-self.init_pos)) > config.rang)
        # done: dead or detect all objects
        done = dead or num_goal == len(config.goal)

        # compute reward
        reward = self.compute_reward(detected, num_goal, dead, done)

        # log info
        info = {}

        if has_collided:
            info['status'] = 'collision'
        

        # quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        # observation = [responses, quad_vel]
        return observation, reward, done, info


    def compute_reward(self, detected, num_goal, dead, done):
        # vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float)
        # speed = np.linalg.norm(vel)
        if dead:
            reward = config.reward['dead']
        elif done:
            reward = config.reward['goal']
        elif num_goal > 0:
            reward = config.reward['goal']* (num_goal/len(config.goal))
        else:
            reward = 0

        return reward

    def detect_image(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene)])

        response = responses[0]
        # image = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # img_rgb = image.reshape(response.height, response.width, 3)
        # image = np.flipud(img_rgb)
        # cv2.imshow("object detection", image)
        # cv2.waitKey()

        airsim.write_file(os.path.normpath('tmp.png'), response.image_data_uint8)
        image = cv2.imread("tmp.png")

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        outs = self.net.forward([layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()])

        return outs, Width, Height

    def detected_obj(self, observation, Width, Height):
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in observation:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])


        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        detected = []
        for i in indices:
            detected.append(class_ids[i])

        num_goal = sum(1 for x, y in zip(detected, config.goal) if x == y)

        return detected, num_goal
        
    def disconnect(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
        print('Disconnected.')