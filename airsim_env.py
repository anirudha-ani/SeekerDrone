import time
import numpy as np
import airsim
import os
import config
import cv2

clockspeed = 1
timeslice = 0.5 / clockspeed


class Env:
    def __init__(self):
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        with open("yolov3.txt", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

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
        # self.client.moveByVelocityBodyFrameAsync(0, 0, -0.3, 1).join()
        self.client.simPause(True)

        detected = self.detected_obj()

        return detected

    def take_action(self, action):
        if action == 0:#w
            self.client.moveByVelocityBodyFrameAsync(1, 0, 0, 1).join()
        elif action == 1:#a
            self.client.rotateByYawRateAsync(-20,1).join()
        # elif action == 2:#s
        #     self.client.moveByVelocityBodyFrameAsync(-1, 0, 0, 1).join()
        elif action == 2:#d
            self.client.rotateByYawRateAsync(20,1).join()


    def step(self, action, goal):
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
        detected = self.detected_obj()

        # get quadrotor states
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position

        # dead: too far from original point or collided #todo
        dead = (sum(map(lambda i : i * i, quad_pos-self.init_pos)) > config.rang) or collided
        # done: dead or detect all objects
        done = dead or (detected[np.argmax(goal[0])][-1] != 0)

        # compute reward
        reward = self.compute_reward(detected, goal, dead, done)

        # log info
        info = {}

        # if has_collided:
        #     info['status'] = 'collision'
        
        return detected, reward, done, info


    def compute_reward(self, detected, goal, dead, done):
        if dead:
            reward = config.reward['dead']
        elif done:
            reward = config.reward['goal']
        else:
            reward = 0
            for i in range(4):
                if detected[i][-1] != 0 and goal[1][i] != 0:
                    reward += config.reward['subgoal']
        # if detected[2][-1] != 0 and goal[2][0] != 0:
        #     reward = 1
        # else:
        #     reward = -1
        return reward



    def detected_obj(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene)])

        response = responses[0]

        airsim.write_file(os.path.normpath('tmp.png'), response.image_data_uint8)
        image = cv2.imread("tmp.png")

        if image is None:
            detected = np.zeros([4,5])
            # detected[:,0:2] = -100
            return detected

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        # net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        observation = self.net.forward([layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()])


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
        detected = np.zeros([4,5])
        # detected[:,0:2] = -100


        for i in indices:
            if class_ids[i] == 11: #stop
                detected[0] = boxes[i] + [confidences[i]]
            elif class_ids[i] == 16: #dog
                detected[1] = boxes[i] + [confidences[i]]
            elif class_ids[i] == 2: #car
                detected[2] = boxes[i] + [confidences[i]]
            elif class_ids[i] == 0: #person
                detected[3] = boxes[i] + [confidences[i]]
            

        return detected
        
    def disconnect(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
        print('Disconnected.')