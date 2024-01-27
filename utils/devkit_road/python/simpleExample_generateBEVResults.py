#!/usr/bin/env python
#
#  THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK
#
#  File: simpleExample_transformTestResults2BEV.py
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

import os, sys
import computingPipeline, transform2BEV

#########################################################################
# test script to process testing data in perspective domain and 
# transform the results to the metric BEV 
#########################################################################

if __name__ == "__main__":
    
    #datasetDir = '/hri/storage/user/rtds/KITTI_Road_Data'
    #outputDir = '/hri/storage/user/rtds/KITTI_Road_Data/test_baseline_bev'

    # check for correct number of arguments.
    if len(sys.argv)<2:
        print "Usage: python simpleExample_generateBEVResults.py  <datasetDir> <outputDir>"
        print "<datasetDir> = base directory of the KITTI Road benchmark dataset (has to contain training and testing), e.g., /home/elvis/kitti_road/"
        print "<outputDir>  = Here the baseline results will be saved, e.g., /home/elvis/kitti_road/results/"
        sys.exit(1)

    # parse parameters
    datasetDir = sys.argv[1]
    assert os.path.isdir(datasetDir), 'Error <datasetDir>=%s does not exist' %datasetDir
    if len(sys.argv)>2:
        outputDir = sys.argv[2]   
    else:
        # default
        outputDir = os.path.join(datasetDir, 'results')
    
    # path2data
    testData_pathToCalib = os.path.join(datasetDir, 'testing/calib')
    outputDir_perspective = os.path.join(outputDir, 'segmentation_perspective_test')
    outputDir_bev = os.path.join(outputDir, 'segmentation_bev_test')
    
    # Run computeBaseline script to generate example classification results on testing set
    # Replace by your algorithm to generate real results
    trainDir = os.path.join(datasetDir, 'training')
    testDir = os.path.join(datasetDir, 'testing')
    computingPipeline.main(testDir, outputDir_perspective)
    
    # Convert baseline in perspective space into BEV space
    # If your algorithm provides results in perspective space,
    # you need to run this script before submission!
    inputFiles = os.path.join(outputDir_perspective, '*.png')
    transform2BEV.main(inputFiles, testData_pathToCalib, outputDir_bev)

    # now zip the contents in the directory 'outputDir_bev' and upload
    # the zip file to the KITTI server


    
