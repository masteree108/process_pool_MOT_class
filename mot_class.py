# USAGE
# python MOTF_process_pool.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video race.mp4

# watch CPU loading on the ubuntu
# ps axu | grep [M]OTF_process_pool.py | awk '{print $2}' | xargs -n1 -I{} ps -o sid= -p {} | xargs -n1 -I{} ps --forest -o user,pid,ppid,cpuid,%cpu,%mem,stat,start,time,command -g {}

# import the necessary packages
from imutils.video import FPS
from multiprocessing import Pool
import numpy as np
import argparse
import imutils
import cv2
import os


class mot_class(): 
#private

    # for saving tracker objects
    __cv_multi_tracker_list = []
    # detected flag
    __detection_ok = False
    # if below variable set to True, this result will not show tracking bbox on the video
    # ,it will show number on the terminal
    __print_number_test_not_tracker = False
    __frame_size_width = 600
    __detect_people_qty = 0

    # initialize the list of class labels MobileNet SSD was trained to detect
    __CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"]
    # can not use class video_capture variable, otherwise this process will crash
    #__vs = 0
    __processor_task_num = []

    def __get_algorithm_tracker(self, algorithm):
        if algorithm == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif algorithm == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif algorithm == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif algorithm == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif algorithm == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif algorithm == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        elif algorithm == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
        elif algorithm == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        return tracker

    def __detect_people_quantity(self, frame, detections, args, w, h):
        # detecting how many person on this frame
        person_num = 0
        bboxes = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:
                idx = int(detections[0, 0, i, 1])
                label = self.__CLASSES[idx]
                #print("label:%s" % label)
                if self.__CLASSES[idx] != "person":
                    continue
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                bb = (startX, startY, endX, endY)
                bboxes.append(bb)
                #print(bb)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                #print("label:%s" % label)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                #self.__init_tracker(person_num, bb, frame)
                person_num = person_num + 1 
        return person_num, bboxes


    def __init_cv_multi_tracker(self, frame, bboxes, detect_people_qty, using_processor_qty):
        # it should brings (left, top, width, height) to tracker.init() function
        # parameters are left, top , right and bottom in the box 
        # so those parameters need to minus like below to get width and height 
        left_num = detect_people_qty % using_processor_qty
        process_num = int(detect_people_qty / using_processor_qty)
        processor_task_num = []                    
        process_num_ct = 0                         
        #print("bboxes:")                          
        #print(bboxes)                             
        for i in range(using_processor_qty):       
            task_ct = 0                            
            tracker = cv2.MultiTracker_create()    
            for j in range(process_num_ct, process_num_ct + process_num):
                #print("j:%d" % j)                 
                bbox =(bboxes[j][0], bboxes[j][1] ,abs(bboxes[j][0]-bboxes[j][2]), abs(bboxes[j][1]-bboxes[j][3]))
                #print("bbox:")                                                                                                                         
                #print(bbox)                       
                tracker.add(self.__get_algorithm_tracker("CSRT"), frame, bbox) 
                task_ct = task_ct + 1              
                process_num_ct = process_num_ct + 1
            self.__cv_multi_tracker_list.append(tracker)  
            processor_task_num.append(task_ct)     
        if left_num != 0:                          
            counter = 0                            
            k = detect_people_qty - using_processor_qty * process_num
            for k in range(k, k+left_num):         
                #print("k:%d" % k)                 
                bbox =(bboxes[k][0], bboxes[k][1] ,abs(bboxes[k][0]-bboxes[k][2]), abs(bboxes[k][1]-bboxes[k][3]))
                self.__cv_multi_tracker_list[counter].add(get_algorithm_tracker("CSRT"),frame , bbox)
                processor_task_num[counter] = processor_task_num[counter] + 1
                counter = counter + 1       
        #print("processor_task_number:")    
        #print(processor_task_num)          
        return processor_task_num    

# public    
    def __init__(self, args):

        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

         # initialize the video stream and output video writer
        print("[INFO] starting video stream...")
        vs = cv2.VideoCapture(args["video"])
    
        # step 1. grab the frame dimensions and convert the frame to a blob
        # step 2. detecting how many people on this frame
        # step 1:
        (grabbed, frame) = vs.read()
        frame = imutils.resize(frame, width=self.__frame_size_width)
        (h, w) = frame.shape[:2]
        print((h,w))
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
        net.setInput(blob)
        detections = net.forward()
        # step 2:
        self.__detect_people_qty = 0
        if self.__print_number_test_not_tracker == False:
            self.__detect_people_qty, bboxes= self.__detect_people_quantity(frame, detections, args, w, h)
            if self.__detect_people_qty >= (os.cpu_count()-1):
                using_processor_qty = os.cpu_count()-1
                self.__processor_task_num = self.__init_cv_multi_tracker(frame, bboxes, self.__detect_people_qty, using_processor_qty)
            else:       
                using_processor_qty = self.__detect_people_qty
                self.__processor_task_num  = self.__init_cv_multi_tracker(frame, bboxes, self.__detect_people_qty, using_processor_qty)

            print("detect_people_qty: %d" % self.__detect_people_qty) 
            print("processor_task_num") 
            print(self.__processor_task_num) 
        else:
            self.__detection_ok = True

        # start the frames per second throughput estimator
        self.__fps = FPS().start()
        self.__now_frame = frame 
        '''
        # for testing, only shows peolpe who have been detected
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        '''

    def start_tracker(self, input_data):    
        bboxes_org = []               
        bboxes_transfer = []          
        n_frame = 0                   
        n_tracker = 1                 
                                  
        tl_num = input_data[n_tracker]                                                                                                                              
        #print("start_tracker, track_list[%d]" % tl_num)
                                  
        ok, bboxes_org = self.__cv_multi_tracker_list[tl_num].update(input_data[n_frame])
        #print(bboxes_org)            
        for box in bboxes_org:        
            startX = int(box[0])                                      
            startY = int(box[1])      
            endX = int(box[0] + box[2])           
            endY = int(box[1] + box[3])
            bbox = (startX, startY, endX, endY)   
            bboxes_transfer.append(bbox)
            #print(bbox)              
        return bboxes_transfer   

    # for pool testing 
    def map_test(self, i):
        print(i)

    # tracking person on the video
    def tracking(self, args):
        vs = cv2.VideoCapture(args["video"])
        if self.__print_number_test_not_tracker == False:
            if self.__detect_people_qty >= (os.cpu_count()-1):
                pool = Pool(os.cpu_count()-1)
            else:
                pool = Pool(processes = self.__detect_people_qty)
        else:
            pool = Pool(3)

        # loop over frames from the video file stream
        while True:
            
	    # grab the next frame from the video file
            if self.__detection_ok == True:
                (grabbed, frame) = vs.read()
                #print("vs read ok")
	        # check to see if we have reached the end of the video file
                if frame is None:
                    break
            else:
                frame = self.__now_frame
                self.__detection_ok = True
        
            frame = imutils.resize(frame, width=self.__frame_size_width)
            if self.__print_number_test_not_tracker == True:
                print("map_test...")
                pool.map(self.map_test, [1,2,3])
                print("map_test ok")
            else:
                input_data = []
                for i in range(self.__detect_people_qty):
                    input_data.append([])
                    input_data[i].append(frame)
                    input_data[i].append(i)

                # can not use map_async,otherwise it will not wait all trackers to finish the job,
                # it will just executing print("before operating cv2") directly
                #pool_output = pool.map_async(start_tracker, input_data)     
                #pool.close()
                #pool.join()                                                                                                                                        

                pool_output = pool.map(self.start_tracker, input_data)
                #print(pool_output)
     
                #print("before operating cv2")
                #print("len(pool_output):%d" % len(pool_output))
                for i in range(len(pool_output)):
                    #print(pool_output[i][0])
                    #print(box)
                    for j in range(self.__processor_task_num[i]):
                        #print(pool_output[i][j])
                        (startX, startY, endX, endY) = pool_output[i][j]
                        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
                        cv2.putText(frame, "preson", (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            #print("before imshow")
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            #if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # update the FPS counter
            self.__fps.update()

        # stop the timer and display FPS information
        self.__fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(self.__fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.__fps.fps()))

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.release()
        pool.close()
        pool.join()

