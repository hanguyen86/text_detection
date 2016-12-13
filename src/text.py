"""
Scene Text Detection and Recognition
Licence: BSD
Author : Hoang Anh Nguyen
"""

import cv2
import numpy as np
import argparse, sys

#--------------------------------------------------------
#--------------------------------------------------------
# Class provide an interface to detect texts in images
class TextDetector:
    
    def __init__(self, image_name):
        # read image file and convert it to grayscale
        self.origin_image = cv2.imread(image_name)
        self.gray_image   = cv2.cvtColor(self.origin_image,
                                         cv2.COLOR_BGR2GRAY)
        
        self.ER_GROUPING_ORIENTATION = cv2.text.ERGROUPING_ORIENTATION_ANY
        
        # load pre-trained ER classifiers
        fileERClassifier1 = cv2.text.loadClassifierNM1(
            './data/trained_classifierNM1.xml')
        self.ERClassifier1 = cv2.text.createERFilterNM1(fileERClassifier1,
                                                        16, 0.00015, 0.13,
                                                        0.2, True, 0.1)
        fileERClassifier2 = cv2.text.loadClassifierNM2(
            './data/trained_classifierNM2.xml')
        self.ERClassifier2 = cv2.text.createERFilterNM2(fileERClassifier2,
                                                        0.5)
        
        # store all found regions
        self.regions = []
        
    #--------------------------------------------------------
    # Main methods
    #--------------------------------------------------------
    
    # Extract text locations in the image using a ER method
    # input:    showResult
    # return:   array of text regions
    def extract(self, showResult = False):
        
        # Extract channels to be processed individually
        channels = cv2.text.computeNMChannels(self.origin_image)
        # Append negative channels to detect ER- (bright regions over dark background)
        cn = len(channels) - 1
        for c in range(0, cn):
            channels.append((255 - channels[c]))

        # Apply the default cascade classifier to each independent channel
        # (could be done in parallel)
        print("Extracting Class Specific Extremal Regions from "
              + str(len(channels)) + " channels ...")
        for channel in channels:
            regions = cv2.text.detectRegions(channel, 
                                             self.ERClassifier1,
                                             self.ERClassifier2)

            if self.ER_GROUPING_ORIENTATION == cv2.text.ERGROUPING_ORIENTATION_ANY:
                rects = cv2.text.erGrouping(self.origin_image,
                                            channel,
                                            [r.tolist() for r in regions],
                                            cv2.text.ERGROUPING_ORIENTATION_ANY,
                                            './data/trained_classifier_erGrouping.xml',
                                            0.7)
            else:
                rects = cv2.text.erGrouping(self.origin_image,
                                            channel,
                                            [r.tolist() for r in regions])
            # save regions
            [self.regions.append(rect) for rect in rects]
            
        if showResult:
            return self.showRegions()
        
        return None
    
    # perform OCR recognition on image: BeamSearchDecoder of HMMDecoder
    # input:  image
    # output: string
    def recognize(self, methodId):
        methodName = [
            'HMMDecoderParser',
            'BeamSearchCNNParser',
            'TesseractParser'
        ][methodId - 1]
        
        ocrer = eval(methodName)()
        return ocrer.parse(self.gray_image)
    
    # draw detected text regions
    def showRegions(self):
        output = self.origin_image.copy()
        for r in range(0, np.shape(self.regions)[0]):
            rect = self.regions[r]
            cv2.rectangle(output,
                          (rect[0],rect[1]),
                          (rect[0]+rect[2],
                           rect[1]+rect[3]),
                          (0, 255, 0), 2)
            cv2.rectangle(output,
                          (rect[0],rect[1]),
                          (rect[0]+rect[2],
                           rect[1]+rect[3]),
                          (255, 0, 0), 1)
        return output
    
#--------------------------------------------------------
#--------------------------------------------------------
# Class provide an interface to perform OCR
class OCRParser:
    
    def __init__(self):
        self.ocrer = None
    
    def parser(self, image):
        return self.ocrer.run(image, 0.5)

# class provides an interface for OCR using Hidden Markov Models
# http://docs.opencv.org/3.1.0/d0/d74/classcv_1_1text_1_1OCRHMMDecoder.html
class HMMDecoderParser(OCRParser):
    
    def __init__(self):
        self.vocabulary = 
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.ocrer = cv2.text.OCRBeamSearchDecoder_create(
            loadOCRHMMClassifierCNN("OCRHMM_knn_model_data.xml.gz"),
            self.vocabulary,
            cv2.cv.Load("OCRHMM_transitions_table.xml"),
            np.eye(62, 62, dtype='float')
        )

# class provides an interface for OCR using Beam Search algorithm
# http://docs.opencv.org/3.1.0/da/d07/classcv_1_1text_1_1OCRBeamSearchDecoder.html
class BeamSearchCNNParser(OCRParser):
    
    def __init__(self):
        self.vocabulary = 
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.ocrer = cv2.text.OCRBeamSearchDecoder_create(
            loadOCRBeamSearchClassifierCNN("OCRBeamSearch_CNN_model_data.xml.gz"),
            self.vocabulary,
            cv2.cv.Load("OCRHMM_transitions_table.xml"),
            np.eye(62, 62, dtype='float')
        )

# class provides an interface with the tesseract-ocr API (v3.02.02) in C++
# http://docs.opencv.org/3.1.0/d7/ddc/classcv_1_1text_1_1OCRTesseract.html
class TesseractParser(OCRParser):
    
    def __init__(self):
        self.ocrer = cv2.text.OCRTesseract_create()
        
#--------------------------------------------------------
#--------------------------------------------------------
def main(argv):
    # Define argument list. Example:
    # python text.py -m 1
    #                -i test/scenetext_segmented_word04.jpg
    #                -o .
    parser = argparse.ArgumentParser(description='Scene Text Detection and Recognition')
    parser.add_argument('-t','--task',
                        help="""Specify method:
                        1: Detection using Extremal Region Filter,
                        2: OCR Recognition using Tesseract
                        """,
                        required=True)
    parser.add_argument('-m','--method',
                        help="""Method for OCR:
                        1: HMMDecoder,
                        2: BeamSearchCNN,
                        3: Tesseract
                        """,
                        required=True)
    parser.add_argument('-i','--input',
                        help='Input image',
                        required=True)
    parser.add_argument('-o','--output',
                        help='Ouput location',
                        required=True)
    args = vars(parser.parse_args())
    
    # extract arguments
    task = int(args['task'])
    method = int(args['method'])
    detector = TextDetector(args['input'])
    
    if task == 1:
        output = detector.extract(True)
        cv2.imwrite(args['output'] + '/output.jpg', output)
        print("Output saved!")
    elif task == 2:
        print(detector.recognize(args['method']))
    else:
        print("Invalid task: " + args['task'])
        return
    
if __name__ == '__main__':
    main(sys.argv)
