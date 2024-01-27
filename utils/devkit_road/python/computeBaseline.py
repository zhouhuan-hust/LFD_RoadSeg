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
import os, sys, cv2
import cv2 # OpenCV

class dataStructure: 
    '''
    All the defines go in here!
    '''
    
    cats = ['um_lane', 'um_road', 'umm_road', 'uu_road']
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
def main(train_dir, test_dir, outputDir):
    '''
    main method of computeBaseline
    :param train_dir: directory of training data (has to contain ground truth: gt_image_2), e.g., /home/elvis/kitti_road/training
    :param test_dir: directory with testing data (has to contain images: image_2), e.g., /home/elvis/kitti_road/testing
    :param outputDir: directory where the baseline results will be saved, e.g., /home/elvis/kitti_road/test_baseline_perspective
    '''


    trainData_path_gt = os.path.join(train_dir, dataStructure.trainData_subdir_gt)
    
    print "Computing category specific location potential as a simple baseline for classifying the data..."
    print "Using ground truth data from: %s" % trainData_path_gt
    print "All categories = %s" %dataStructure.cats
    
    # Loop over all categories
    for cat in dataStructure.cats:
        cat_tags = cat.split('_')
        print "Computing on dataset: %s for class: %s" %(cat_tags[0],cat_tags[1])
        trainData_fileList_gt = glob(os.path.join(trainData_path_gt, cat + '*' + dataStructure.gt_end))
        trainData_fileList_gt.sort()
        assert len(trainData_fileList_gt)>0, 'Error: Cannot find ground truth data in %s' % trainData_path_gt
        
        # Compute location potential
        locationPotential = np.zeros(dataStructure.imageShape_max, 'f4')
        # Loop over all gt-files for particular category
        for trainData_fileName_gt in trainData_fileList_gt:
            
            full_gt = cv2.imread(trainData_fileName_gt, cv2.CV_LOAD_IMAGE_UNCHANGED)
            #attention: OpenCV reads in as BGR, so first channel has road GT
            trainData_file_gt =  full_gt[:,:,0] > 0
            #validArea = full_gt[:,:,2] > 0
            
            assert locationPotential.shape[0] >= trainData_file_gt.shape[0], 'Error: Y dimension of locationPotential is too small: %d' %trainData_file_gt.shape[0]
            assert locationPotential.shape[1] >= trainData_file_gt.shape[1], 'Error: X dimension of locationPotential is too small: %d' %trainData_file_gt.shape[1]
            
            locationPotential[:trainData_file_gt.shape[0], :trainData_file_gt.shape[1]] += trainData_file_gt
        
        # Compute prop
        locationPotential = locationPotential/len(trainData_fileList_gt)
        locationPotential_uinit8 = (locationPotential*255).astype('u1')
        
        print "Done: computing location potential for category: %s." %cat
        
        if not os.path.isdir(outputDir):
            os.makedirs(outputDir)
    
        testData_fileList_im2 = glob(os.path.join(test_dir, dataStructure.testData_subdir_im2, cat_tags[0] + '_*'+ dataStructure.im_end))
        testData_fileList_im2.sort()
        
        print "Writing location potential as perspective probability map into %s." %outputDir
        
        for testData_file_im2 in testData_fileList_im2:
            # Write output data (same format as images!)
            fileName_im2 = testData_file_im2.split('/')[-1]
            ts_str = fileName_im2.split(cat_tags[0])[-1]
            fn_out = os.path.join(outputDir, cat + ts_str)
            cv2.imwrite(fn_out, locationPotential_uinit8)
            
        print "Done: Creating perspective baseline."


if __name__ == "__main__":
    
    # check for correct number of arguments.
    if len(sys.argv)!=4:
        print "Usage: python coomputeBaseline.py <TrainDir> <TestDir> <OutputDir> "
        print "<TrainDir> = directory of training data (has to contain ground truth: gt_image_2), e.g., /home/elvis/kitti_road/training"
        print "<TestDir> = directory with testing data (has to contain images: image_2), e.g., /home/elvis/kitti_road/testing"
        print "<OutputDir>  = directory where the baseline results will be saved, e.g., /home/elvis/kitti_road/test_baseline_perspective"
        sys.exit(1)
    
    # parse parameters
    trainDir = sys.argv[1]
    testDir = sys.argv[2]
    outputDir = sys.argv[3] # Directory for saveing the output data
    
    # Excecute main fun
    main(trainDir, testDir, outputDir)
        
