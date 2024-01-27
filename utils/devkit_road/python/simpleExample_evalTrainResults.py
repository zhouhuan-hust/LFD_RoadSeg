#!/usr/bin/env python
#
#  THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK
#
#  File: simpleExample_evalTrainResults.py
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
import computeBaseline, evaluateRoad

#########################################################################
# test script to evaluate training data in perspective domain
#########################################################################

if __name__ == "__main__":
    
    #datasetDir = '/hri/storage/user/rtds/KITTI_Road_Data'
    #outputDir = '/hri/recordings/KITTI/road_dataset/'
    
    # check for correct number of arguments.
    if len(sys.argv)<2:
        print "Usage: python simpleExample_evalTrainResults.py  <datasetDir> <outputDir>"
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
    
    
    # Run computeBaseline script to generate example classification results on training set
    trainDir = os.path.join(datasetDir, 'training')
    outputDir_perspective = os.path.join(outputDir, 'baseline_perspective_train')
    computeBaseline.main(trainDir, trainDir, outputDir_perspective)
    
    # Toy example running evaluation on perspective train data
    # Final evaluation on server is done in BEV space and uses a 'valid_map'
    # indicating the BEV areas that are invalid
    # (no correspondence in perspective space)
    evaluateRoad.main(outputDir_perspective, trainDir)
