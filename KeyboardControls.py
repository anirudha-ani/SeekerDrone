import keyboard  # using module keyboard
import time
import os

import airsim #pip install airsim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# OpenCV YOLO implementation 


class RequiredParameters():
    def __init__(self):
        self.image = "py3.png"
        self.config = "yolov3.cfg"
        self.weights= "yolov3.weights"
        self.classes = "yolov3.txt"

args = RequiredParameters()
# args.image = "dog.jpg"
# args.config = "yolov3.cfg"
# args.weights= "yolov3.weights"
# args.classes = "yolov3.txt"

def get_output_layers(net):
    
    layer_names = net.getLayerNames()

    # print(layer_names)
    
    # for i in net.getUnconnectedOutLayers():
    #     print(">>", i)
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    

# Keyboard implementation
client.moveByVelocityBodyFrameAsync(0, 0, -1, 1).join()

while True:  # making a loop
    keyboardEntry = input("Enter a key :")
    try:  # used try so that if user pressed other than the given key error will not be shown
        if keyboardEntry == "w":  # if key 'q' is pressed 
            print('You Pressed W Key!')
            # client.takeoffAsync().join()
            # client.moveToPositionAsync(10, 0, 0, 5).join()
            client.moveByVelocityBodyFrameAsync(5, 0, 0, 1).join()
            
            time.sleep(.15)
              # finishing the loop
        elif keyboardEntry == "a":  # if key 'q' is pressed 
            print('You Pressed a Key!')
            # client.moveByVelocityBodyFrameAsync(0, 10, 0, 3).join()
            client.rotateByYawRateAsync(-20,1).join()
            # client.rotateToYawAsync(90,5,1).join()
            # client.moveByRollPitchYawZAsync(0,0,.7, 1, 5).join()
            time.sleep(.15)
              # finishing the loop
        elif keyboardEntry == "s":  # if key 'q' is pressed 
            print('You Pressed s Key!')
            
            client.moveByVelocityBodyFrameAsync(-5, 0, 0, 1).join()
            
            time.sleep(.15)
              # finishing the loop
        elif keyboardEntry == "d":  # if key 'q' is pressed 
            print('You Pressed d Key!')
            # client.moveByVelocityBodyFrameAsync(0, -10, 0, 3).join()
            client.rotateByYawRateAsync(20,1).join()
            # client.rotateByYawRateAsync(0.25, 5).join()
            time.sleep(.15)
              # finishing the loop
        elif keyboardEntry == "u":  # if key 'q' is pressed 
            print('You Pressed u Key!')
            # client.takeoffAsync().join()
            # client.moveToZAsync(-5,1).join()
            client.moveByVelocityBodyFrameAsync(0, 0, -1, 1).join()
            time.sleep(.15)
              # finishing the loop
        elif keyboardEntry == "j":  # if key 'q' is pressed 
            print('You Pressed j Key!')
            client.moveByVelocityBodyFrameAsync(0, 0, 1, 1).join()
            time.sleep(.15)
              # finishing the loop
        elif keyboardEntry == "c":
            responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis),
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True),
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective),
            airsim.ImageRequest("0", airsim.ImageType.Scene),
            airsim.ImageRequest("0", airsim.ImageType.Infrared)])
            print('Retrieved images: %d', len(responses))

            # do something with the images
            index = 0
            for response in responses:
                if response.pixels_as_float:
                    print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                    airsim.write_pfm(os.path.normpath('py' + str(index)+ '.pfm'), airsim.get_pfm_array(response))
                else:
                    print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                    airsim.write_file(os.path.normpath('py' + str(index)+ '.png'), response.image_data_uint8)
                index+= 1
            

            
            image = cv2.imread(args.image)

            Width = image.shape[1]
            Height = image.shape[0]
            scale = 0.00392

            classes = None

            with open(args.classes, 'r') as f:
                classes = [line.strip() for line in f.readlines()]

            COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

            net = cv2.dnn.readNet(args.weights, args.config)

            blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

            net.setInput(blob)

            outs = net.forward(get_output_layers(net))

            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4


            for out in outs:
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

            for i in indices:
                # i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

            cv2.imshow("object detection", image)
            cv2.waitKey() 
                
            cv2.imwrite("object-detection.jpg", image)
            cv2.destroyAllWindows()

        elif keyboardEntry == "x":  # if key 'q' is pressed 
            print('You Pressed x Key!')
            break  # finishing the loop
    except:
        break