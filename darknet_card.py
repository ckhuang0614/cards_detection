from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread #, enumerate
from queue import Queue


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    parser.add_argument("--cards_file", default="./cards.ini",
                        help="path to cards file")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))
    if not os.path.exists(args.cards_file):
        raise(ValueError("Invalid card file path {}".format(os.path.abspath(args.cards_file))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def video_capture(frame_queue, darknet_image_queue, cv_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame_resized)
        cv_queue.put(frame_resized)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        darknet_image_queue.put(darknet_image)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue, location_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        #print("FPS: {}".format(fps))
        print("detections: {}".format(detections))
        location_queue.put(detections)
        darknet.print_detections(detections, args.ext_output)
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (width, height))
    while cap.isOpened():
        frame_resized = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        if frame_resized is not None:
            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # for get cv test image 
            #image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            if args.out_filename is not None:
                video.write(image)
            if not args.dont_show:
                cv2.imshow('Inference', image)
            if cv2.waitKey(fps) == 27:
                break
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    
import numpy as np

def block_similarity(img1,cut_img,orb,des1):
    # 初始化ORB檢測器
    kp2, des2 = orb.detectAndCompute(cut_img, None)
    # 提取並計算特徵點
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # knn篩選結果
    try:
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
        #牌背相似程度
        likely_num = 0.6
        good = [m for (m, n) in matches if m.distance < likely_num * n.distance]
        similary = len(good) / (len(matches)+0.0000001)
    except :
        similary=0
    return similary

def img_cut(img1,img2,area,strid,orb,des1): 
    t=x=y=r=0
    for i in range(0,int((564-area)/strid)):
        for j in range(0,int((774-area)/strid)):
            r +=1
            cut_img = img2[x:(x+area),y:(y+area)]
            similary = block_similarity(img1,cut_img,orb,des1)
            if similary > 0:
                t +=1
            x = j*strid
        y = i*strid
    return((t/r))
    
def cvCrossFilter(cv_queue, action_queue, gError, BG, card_back):
    while cap.isOpened():
        cvframe_resized = cv_queue.get()
        print("cv_queue.qsize", cv_queue.qsize())
        if cvframe_resized is not None:
            # todo: check Cross+Filter, if error put action_queue.put(error_info)
            print("check Cross+Back!")
            #action_queue.put("Cross+Filter Error!")
            
            if len(cvframe_resized) > 0:
                cvframe_resized = cv2.cvtColor(cvframe_resized, cv2.COLOR_RGB2GRAY)
                print(cvframe_resized.shape)

                tStart = time.time()#計時開始
                print("cvframe_resized" , len(cvframe_resized))
                img2 = cv2.resize(cvframe_resized,None,fx=0.5,fy=0.5)
                print("cvframe_resized" ,len(img2))
                img2 = img2[35:175, 20:180]
                cv2.imwrite("final_img2.png",img2)
                
                #以底板製作Mask
                diff_img=cv2.subtract(img2,BG,BG_INV)
                ret,diff = cv2.threshold(diff_img,100,1,cv2.THRESH_BINARY)
                #cv2.imwrite("final_diff.png",diff)
                
                diff_num = diff.sum()
                print(diff_num)

                if diff_num > 0:
                    print("cvframe → touch line!")          
                else:
                    print("cvframe → OK")   
                print("card cv inspect time:%.2f"%(time.time() - tStart))

#                 S1, S2 = card_back(card_back,cvframe_resized,area,strid,orb,des1)
                S1 = img_cut(card_back,cvframe_resized,area,strid,orb,des1)
                #特徵翻轉
                card_back = cv2.flip(card_back, 0)
                S2 = img_cut(card_back,cvframe_resized,area,strid,orb,des1)

                #整張影像中牌背面積比例
                print("card_back_area = %.3f%% " % ((S1+S2)*100))
                if S1+S2 > 0:  
                    print("card_back → NG")
                else:
                    print("card_back → OK")
                
                
                
    cap.release()

# function to find if given point lies inside a given rectangle or not. 
def FindPoint(x1, y1, x2, y2, x, y): 
    if (x > x1 and x < x2 and y > y1 and y < y2): 
        return True
    else: 
        return False
    
def actionLocation(location_queue, action_queue, gStatus, gCards, gTask_info, gBlock_info, gError):
    
    summary_info = {'status': 0, 'err-message': []}
    
    while cap.isOpened():
        location = location_queue.get()
        if location is not None:
            
            print("gStatus==>", gStatus)
            
            if len(location) == 0:
                # all blocks have no cards
                if gError == 0 and gStatus == 2:
                    gStatus = 0
                    print("Change gStatus==>", gStatus)
                    summary_info['status'] = 0
                    summary_info['err-message'] = []
                    action_queue.put(summary_info)
                    # Inital Random gTask_info
                    random.shuffle(gCards)
                    print("Random gCards", gCards)
                    gTask_info['1st'] = gCards[0]
                    gTask_info['2nd'] = gCards[1]
                    gTask_info['3rd'] = gCards[2]
                    gTask_info['4th'] = gCards[3]
                    print("Initial Task_info", gTask_info)
            else:
                # first card into webcam
                if gStatus == 0:
                    gStatus = 1
                    print("Change gStatus==>", gStatus)
                    summary_info['status'] = 1
                    summary_info['err-message'] = []
                    action_queue.put(summary_info)


                block_elements = {'1st': [], '2nd': [], '3rd': [], '4th': []}
                # check all blocks have real cards
                for loc in location:
                    for bloc in gBlock_info:
                        print("block card=", bloc, loc[0], loc[1], loc[2][0], loc[2][1])
                        mybloc = gBlock_info.get(bloc)

                        if FindPoint(mybloc[0], mybloc[1], mybloc[0]+mybloc[2], mybloc[1]+mybloc[3], loc[2][0], loc[2][1]):
                            block_element = block_elements.get(bloc)
                            if len(block_element) > 0:
                                #for element in block_element:
                                pop_obj = -1
                                for i in range(len(block_element)):
                                    element = block_element[i]
                                    print(element[0], loc[0], float(loc[1]) , float(element[1]))
                                    if element[0] == loc[0] and float(loc[1]) > float(element[1]):
                                        print("modify block unique card!", loc[1], loc[2][0], loc[2][1], loc[2][2], loc[2][3] )
                                        pop_obj = i
                                if pop_obj >= 0:
                                    block_element.pop(i)

                                block_element.append(loc)
                                print("final block_element!", block_element )

                            else:
                                print("add block unique card!", bloc, loc[0], loc[1])
                                block_elements.get(bloc).append(loc)
                        else:
                            print("FindPoint else!")

                print("block_elements", block_elements)

                # check all blocks have real cards
                allTask_Done = 0
                summary_info['err-message'] = []
                for bloc in block_elements:
                    mybloc = block_elements.get(bloc)

                    # mutiple cards
                    if len(mybloc) > 1:
                        print(bloc, "Task_Fail!", "muti-cards= ", mybloc)
                        summary_info['status'] = 1
                        summary_info['err-message'].append("Task_Fail!-1")
                    # check task block card
                    elif len(mybloc) == 1:
                        if gTask_info.get(bloc) == mybloc[0][0]:
                            print(bloc, "Task_OK!", mybloc[0][0])
                            allTask_Done += 1
                        else:
                            print(bloc, "Task_Fail! target=", gTask_info.get(bloc), 'webcam=', mybloc[0][0])
                            summary_info['status'] = 1
                            summary_info['err-message'].append("Task_Fail!-2")
                    # check task block card
                    else:
                        print(bloc, "Task_KeepGoing!")
                        pass    

                if gError == 0 and gStatus == 1 and allTask_Done == 4:
                    gStatus = 2
                    print("Clear Table!")
                    print("Change gStatus==>", gStatus)
                    summary_info['status'] = 2
                    summary_info['err-message'] = []

                action_queue.put(summary_info)

        else:
            pass
            
    cap.release()


def showDashboard(action_queue, gStatus, gTask_info):
    while cap.isOpened():
        action = action_queue.get()
        if action is not None:
            # todo: show action
            print("show_action!", action)
            
            

    cap.release()

            
if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    
    cv_queue = Queue()
    location_queue = Queue()
    action_queue = Queue()
        
    # gStatus: 0-> Initial , 1-> Running, 2-> Reset
    gStatus = 0
    # gCards
    #gCards = ['6h', 'Qd', '10d', '9d', '8d', '2d', 'Kc', '5c', '2c', '3c']
    gCards = []
    # gTask_info
    #gTask_info = {'1st': '6h', '2nd': 'Qd', '3rd': '10d', '4th': '9d'}
    gTask_info = {'1st': '', '2nd': '', '3rd': '', '4th': ''}
    # gBlock_info
    gBlock_info = {'1st': [0, 0, 208, 208], '2nd': [208, 0, 208, 208], '3rd': [0, 208, 208, 208], '4th': [208, 208, 208, 208]}
    
    #for cvCrossFilter check
    gError = 0
    
    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    
    # Inital gCards
    f = open(args.cards_file, 'r') # 開啟並讀取檔案
    line = f.readline()
    gCards = [x.strip() for x in line.split(",")]
    print("Cards:", gCards)
    f.close()
    
    # Inital Random gTask_info
    random.shuffle(gCards)
    print("Random gCards", gCards)
    gTask_info['1st'] = gCards[0]
    gTask_info['2nd'] = gCards[1]
    gTask_info['3rd'] = gCards[2]
    gTask_info['4th'] = gCards[3]
    print("Initial Task_info", gTask_info)

    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    
#     print("CAP_PROP_FPS", cap.get(cv2.CAP_PROP_FPS))
#     cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # depends on fourcc available camera
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     cap.set(cv2.CAP_PROP_FPS, 5)
#     k = cv2.waitKey(100)
    print("CAP_PROP_FRAME_WIDTH", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("CAP_PROP_FRAME_HEIGHT", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("CAP_PROP_FPS", cap.get(cv2.CAP_PROP_FPS))
    
    #BG空版影像
    BG = cv2.imread('background.png',0)
    #BG = cv2.cvtColor(BG, cv2.COLOR_BGR2GRAY)
    print("BG" ,len(BG))
    BG = cv2.resize(BG,None,fx=0.5,fy=0.5)
    print("BG" ,len(BG))
    BG = BG[35:175, 20:180]
    ret,BG_INV = cv2.threshold(BG,127,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite("bg_cut.png",BG)
    #allFileList = os.listdir('./line_check/')
    
    card_back = cv2.imread('cardback.png',0)
    ret,card_back = cv2.threshold(card_back,180,255,cv2.THRESH_BINARY)
    area=100
    strid=50
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(card_back, None)
    
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue, cv_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue, location_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()
    
    Thread(target=cvCrossFilter, args=(cv_queue, action_queue, gError, BG, card_back)).start()
    Thread(target=actionLocation, args=(location_queue, action_queue, gStatus, gCards, gTask_info, gBlock_info, gError)).start()
    Thread(target=showDashboard, args=(action_queue, gStatus, gTask_info)).start()
    
    

