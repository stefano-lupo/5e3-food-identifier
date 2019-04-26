import os
from typing import List, Dict
import random
import shutil

import numpy as np

from darknet_sample import DarknetSample
from sample_name import SampleName

# Fix seed to reproduce training / eval sets
random.seed(1)
np.random.seed(1)

LABELS_DIR = "../labels"
DATASET_DIR = "../dataset"
IMAGES_DIR = "../images"
EVAL_DIR = os.path.join(DATASET_DIR, "eval")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")

# Basic eval split for now, cross val later
EVALUATION_SPLIT = 0.2

def generateSamples(menuItemDirs: List[str]) -> List[DarknetSample]:
    darknetSamples: List[DarknetSample] = []
    for menuItemDir in menuItemDirs:
        menuItemDir = os.path.join(LABELS_DIR, menuItemDir)
        for labelsFile in os.listdir(menuItemDir):
            print(labelsFile)
            sampleName: SampleName = SampleName.fromFilename(labelsFile)
            darknetSamples.append(DarknetSample(IMAGES_DIR, LABELS_DIR, sampleName))
    return darknetSamples

def moveAll(files: List[str], destDir: str):
    for fileName in files:
        shutil.move(fileName, destDir)

def splitDataset(samples: List[DarknetSample]) -> (List[DarknetSample], List[DarknetSample]):
    np.random.shuffle(samples)
    numEval = round(len(samples) * EVALUATION_SPLIT)

    return ( samples[numEval:], samples[:numEval])


def writeDataset(samples: List[DarknetSample], dir: str):
    for sample in samples:
         with open(os.path.join(dir, sample.g), 'w') as f:
                f.writelines(darknetSample.getLines())


if __name__ == "__main__":
    
    menuItemDirs: List[str] = os.listdir(LABELS_DIR)
    print("Using menu items: %s" % menuItemDirs)

    darknetSamples: List[DarknetSample] = generateSamples(menuItemDirs)
    trainSamples, evalSamples = splitDataset(darknetSamples)

    # Clean old data directory whether it exists or not
    shutil.rmtree(DATASET_DIR, ignore_errors=True)
    os.mkdir(DATASET_DIR)
    os.mkdir(TRAIN_DIR)
    os.mkdir(EVAL_DIR)

    for sample in trainSamples:
        sample.writeSample(TRAIN_DIR)

    for sample in evalSamples:
        sample.writeSample(EVAL_DIR)
