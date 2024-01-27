#!/usr/bin/env python
#
#  THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK
#
#  Copyright (C) 2013
#  Honda Research Institute Europe GmbH
#  Carl-Legien-Str. 30
#  63073 Offenbach/Main
#  Germany
#
#  UNPUBLISHED PROPRIETARY MATERIAL.
#  ALL RIGHTS RESERVED.
#
#  Authors: Tobias Kuehnl <tkuehnl@cor-lab.uni-bielefeld.de>
#           Jannik Fritsch <jannik.fritsch@honda-ri.de>
#

import numpy as np
from glob import glob
from helper import overlayImageWithConfidence
import os, sys, cv2
import cv2 # OpenCV

class dataStructure: 
    '''
    All the defines go in here!
    '''
    
    cats = ['um_', 'umm_', 'uu_']
    calib_end = '.txt'
    im_end = '.png'
    gt_end = '.png'
    prob_end = '.png'
    eval_propertyList = ['MaxF', 'AvgPrec', 'PRE_wp', 'REC_wp', 'FPR_wp', 'FNR_wp' ] 
    trainData_subdir_gt = 'gt_image_2'
    testData_subdir_im2 = 'image_2'
    imageShape_max = (376, 1242,)

#########################################################################
# function that computes a road/lane classifiction baseline
#########################################################################
def main(test_dir, outputDir):
    '''
    main method of computing pipeline (simple color segmentation on test data)
    :param test_dir: directory with testing data (has to contain images: image_2), e.g., /home/elvis/kitti_road/testing
    :param outputDir: directory where the baseline results will be saved, e.g., /home/elvis/kitti_road/test_baseline_perspective
    '''

    #trainData_path_gt = os.path.join(train_dir, dataStructure.trainData_subdir_gt)
    
#     print "Computing category specific location potential as a simple baseline for classifying the data..."
#     print "Using ground truth data from: %s" % trainData_path_gt
    print "All categories = %s" %dataStructure.cats
    
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    

    type_tag = 'road'
    
    target_colorValue = [128,128,128]
    vis_threshold = 0.7
    vis_channel = 1
    visFlag = False
    
    if visFlag:
        #cv2.namedWindow('cur_image')
        cv2.namedWindow('cur_conf')
    # Loop over all categories
    for cat in dataStructure.cats:
        
        outputfileName = cat + type_tag + '_' + "%s" + dataStructure.prob_end
        
        testData_fileList_im2 = glob(os.path.join(test_dir, dataStructure.testData_subdir_im2, cat + '*'+ dataStructure.im_end))
        testData_fileList_im2.sort()
        
        print "Writing probability map into %s." %outputDir
        
        for testData_file_im2 in testData_fileList_im2:
            # Write output data (same format as images!)
            fileName_im2 = testData_file_im2.split('/')[-1]
            ts_str = fileName_im2.split(cat)[-1].split(dataStructure.im_end)[0]
            cur_image = cv2.imread(testData_file_im2, -1)
            
            # Simple confidence from image
            offset = abs(cur_image[:,:,0].astype('f4')-target_colorValue[0])/128. + abs(cur_image[:,:,1].astype('f4')-target_colorValue[1])/128. + abs(cur_image[:,:,2].astype('f4')-target_colorValue[2])/128
            cur_conf = np.clip(1.-offset/3, 0., 1.)
            
            if visFlag:
                visImage = overlayImageWithConfidence(cur_image, cur_conf, threshold = threshold)
                #cv2.imshow('cur_image', cur_image)
                cv2.imshow('cur_conf', visImage)
                cv2.waitKey(0)
            
            
            # Write output (BEV)
            fn_out = os.path.join(outputDir, outputfileName %ts_str)
            cv2.imwrite(fn_out, (cur_conf*255).astype('u1'))
            print "done saving %s..." %fn_out
                        
        print "Done: Creating results."


if __name__ == "__main__":
    
    # check for correct number of arguments.
    if len(sys.argv)!=3:
        print "Usage: python computingPipeline.py <TestDir> <OutputDir> "      
        print "<TestDir> = directory with testing data (has to contain images: image_2), e.g., /home/elvis/kitti_road/testing"
        print "<OutputDir>  = directory where the baseline results will be saved, e.g., /home/elvis/kitti_road/test_baseline_perspective"
        sys.exit(1)
    
    # parse parameters
    testDir = sys.argv[1]
    outputDir = sys.argv[2] # Directory for saveing the output data
    
    # Excecute main fun
    main(testDir, outputDir)
        
