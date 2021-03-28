import mot_class as mtc
import os              
import argparse

def read_user_input_info():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,                                                                                                            
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-v", "--video", required=True,
        help="path to input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
     
    return args

                
if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    args = read_user_input_info()
    
    # load our serialized model from disk                                   
    mc = mtc.mot_class(args)
    mc.tracking(args)
