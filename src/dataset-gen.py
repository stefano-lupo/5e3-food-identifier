import os
import sys
from typing import List, Dict, Set
from collections import defaultdict
import random
import shutil
import glob

import numpy as np

from darknet_sample import DarknetSample
from sample_name import SampleName

# Fix seed to reproduce training / eval sets
random.seed(10)
np.random.seed(10)


LABELS_DIR = "./labels"
IMAGES_DIR = "./images"
DARKNET_TEMPLATES_DIR = "./templates"
DARKNET_DATA_FILE_TEMPLATE = os.path.join(DARKNET_TEMPLATES_DIR, "obj.data")

TRAINING_FILES_DIR = "./training-files"
DARKNET_CONFIG_DIR = os.path.join(TRAINING_FILES_DIR, "darknet-configs")
DARKNET_BACKUP_DIR = os.path.join(DARKNET_CONFIG_DIR, "backup")

DATASET_DIR = os.path.join(TRAINING_FILES_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
EVAL_DIR = os.path.join(DATASET_DIR, "eval")

TRAIN_FILE = os.path.join(DARKNET_CONFIG_DIR, "train.txt")
EVAL_FILE = os.path.join(DARKNET_CONFIG_DIR, "eval.txt")

DARKNET_NAMES_FILE = os.path.join(DARKNET_CONFIG_DIR, "obj.names")
DARKNET_DATA_FILE = os.path.join(DARKNET_CONFIG_DIR, "obj.data")



# Basic eval split for now, cross val later
EVALUATION_SPLIT = 0.1

def generateSamples(menuItemDirs: List[str], normalize=False) -> List[DarknetSample]:
    darknetSamples: List[DarknetSample] = []
    for menuItemDir in menuItemDirs:
        menuItemDir = os.path.join(LABELS_DIR, menuItemDir)
        for labelsFile in os.listdir(menuItemDir):
            print(labelsFile)
            sampleName: SampleName = SampleName.fromFilename(labelsFile)
            darknetSamples.append(DarknetSample(IMAGES_DIR, LABELS_DIR, sampleName, normalize))
    return darknetSamples

def moveAll(files: List[str], destDir: str):
    for fileName in files:
        shutil.move(fileName, destDir)

def splitDataset(samples: List[DarknetSample]) -> (List[DarknetSample], List[DarknetSample]):
    # np.random.shuffle(samples)

    samplesByDish: Dict[str, List[DarknetSample]] = defaultdict(list)
    for sample in samples:
        samplesByDish[sample.sampleName.menuItemId].append(sample)

    numClasses: int = len(samplesByDish.keys())
    numEvalPerClass: int = round((len(samples) * EVALUATION_SPLIT) / numClasses)

    evalSamples: Set[DarknetSample] = set()
    for id, samplesForId in samplesByDish.items():
        np.random.shuffle(samplesForId)
        evalSamples.update(samplesForId[:numEvalPerClass])

    trainSamples: Set[DarknetSample] = set(samples) - evalSamples

    return trainSamples, evalSamples
    # return (samples[numEval:], samples[:numEval])


def writeDatasetTxtFiles():
    with open(TRAIN_FILE, 'w') as f:
        trainFiles: List[str] = [trainFile for trainFile in glob.glob("%s/*.jpg" % TRAIN_DIR)]
        f.write("\n".join(trainFiles))

    with open(EVAL_FILE, 'w') as f:
        evalFiles: List[str] = [evalFile for evalFile in glob.glob("%s/*.jpg" % EVAL_DIR)]
        f.write("\n".join(evalFiles))

def generateYoloConfig(darknetSamples: List[DarknetSample]):
    classes: Set[int] = set()
    for sample in darknetSamples:
        objects: List[int] = [obj.label for obj in sample.objects]
        classes.update(objects)
    with open(DARKNET_NAMES_FILE, 'w') as f:
        f.write("\n".join([str(c) for c in classes]))
    with open(DARKNET_DATA_FILE_TEMPLATE, 'r') as f:
        template: str = "".join(f.readlines())
        dataFileContents: str = template.format(
            numClasses=len(classes), 
            trainTxt=TRAIN_FILE, 
            evalFile=EVAL_FILE, 
            namesFile=DARKNET_NAMES_FILE,
            backupDir=DARKNET_BACKUP_DIR)
        with open(DARKNET_DATA_FILE, 'w') as f2:
            f2.write(dataFileContents)


if __name__ == "__main__":
    
    normalize = True
    if len(sys.argv) > 1:
        normalize = sys.argv[1]

    menuItemDirs: List[str] = os.listdir(LABELS_DIR)
    print("Using menu items: %s" % menuItemDirs)

    darknetSamples: List[DarknetSample] = generateSamples(menuItemDirs, normalize)
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
    
    writeDatasetTxtFiles()
    generateYoloConfig(darknetSamples)
    
