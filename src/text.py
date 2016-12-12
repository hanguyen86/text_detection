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
        
        self.ER_GROUPING_ORIENTATION = 0#cv2.text.ERGROUPING_ORIENTATION_ANY
        
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
                                            0.5)
            else:
                rects = cv2.text.erGrouping(self.origin_image,
                                        channel,
                                        [r.tolist() for r in regions])
            # save regions
            [self.regions.append(rect) for rect in rects]
            
        if showResult:
            return self.showRegions()
        
        return None
    
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
def main(argv):
    # Define argument list. Example:
    # python text.py -m 1
    #                -i test/scenetext_segmented_word04.jpg
    #                -o .
    parser = argparse.ArgumentParser(description='Scene Text Detection and Recognition')
    parser.add_argument('-m','--method',
                        help="""Specify method:
                        1: Detection using Extremal Region Filter,
                        2: OCR Recognition using Tesseract
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
    method = int(args['method'])
    
    if method != 1 and method != 2:
        print("Invalid method: " + args['method'])
        return
    
    detector = TextDetector(args['input'])
    output = detector.extract(method == 1)
    
    # save output
    cv2.imwrite(args['output'] + '/output.jpg', output)
    print("Output saved!")
    
if __name__ == '__main__':
    main(sys.argv)
