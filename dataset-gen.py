import os
from typing import List, Dict
import random
import shutil

import numpy as np

# Fix seed to reproduce training / eval sets
random.seed(1)
np.random.seed(1)

CATEGORIES = ["Fork", "Knife", "Spoon", "Glass", "Rice", "Chicken"]
MENU_ITEMS = {
    "001": "Chicken and Rice"
}

BBOX_LABELS_DIR = "bbox-labels"
DARKNET_LABELS_DIR = "darknet-labels"
DATASET_DIR = "dataset"
EVAL_DIR = os.path.join(DATASET_DIR, "eval")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")

# Basic eval split for now, cross val later
EVALUATION_SPLIT = 0.2

"""
BBox: [bounding box left X] [bounding box top Y] [bounding box right X] [bounding box bottom Y] [Label]
DN: [Label] [object center in X] [object center in Y] [object width in X] [object width in Y]
"""
def bBoxLineToDnLine(bBoxLine: str) -> str:
    bBoxPieces = bBoxLine.split(" ")
    label = bBoxPieces[4]
    x1, y1, x2, y2 = [int(i) for i in bBoxPieces[:4]]
    
    width = x2 - x1
    height = y2 - y1
    x = x1 + width / 2
    y = y1 + height / 2

    dnPieces = [str(i) for i in [label, x, y, width, height]]
    dnLine = " ".join(dnPieces)

    return dnLine

def bBoxFileToDnFile(filename: str):
    with open(filename) as f:
        # first line is junk in BBox Multiclass
        bBoxLines = [line.strip("\n") for line in f.readlines()[1:]]
        dnLines = [bBoxLineToDnLine(line) for line in bBoxLines]

        return dnLines

def convertBBoxToDarknet():
    for menuItemDir in menuItemsDirs:
        menuItemDir = os.path.join(BBOX_LABELS_DIR, menuItemDir)
        for labelsFile in os.listdir(os.path.join(menuItemDir)):
            labelsFileFullPath = os.path.join(menuItemDir, labelsFile)
            darknetLines = bBoxFileToDnFile(labelsFileFullPath)
            with open(os.path.join(DATASET_DIR, labelsFile), 'w') as f:
                f.writelines("\n".join(darknetLines))

def moveAll(files: List[str], destDir: str):
    for fileName in files:
        shutil.move(fileName, destDir)

def splitDataset():
    outputFiles: List[str] = os.listdir(DATASET_DIR)
    np.random.shuffle(outputFiles)
    numEval = round(len(outputFiles) * EVALUATION_SPLIT)

    evalSet: List[str] = [os.path.join(DATASET_DIR, f) for f in outputFiles[:numEval]]
    trainingSet: List[str] = [os.path.join(DATASET_DIR, f) for f in outputFiles[numEval:]]

    evalSize: int = len(evalSet)
    trainSize: int = len(trainingSet)

    print("Using %d training samples and %d evaluation samples (%.2f)" % (trainSize, evalSize, (evalSize) / (trainSize + evalSize)))

    os.mkdir(EVAL_DIR)  
    os.mkdir(TRAIN_DIR)

    moveAll(evalSet, EVAL_DIR)
    moveAll(trainingSet, TRAIN_DIR)


if __name__ == "__main__":
    
    # Clean old data directory whether it exists or not
    shutil.rmtree(DATASET_DIR, ignore_errors=True)
    os.mkdir(DATASET_DIR)

    menuItemsDirs: List[str] = os.listdir(BBOX_LABELS_DIR)
    menuItems: Dict[str, str] = {id:mi for id, mi in MENU_ITEMS.items() if id in menuItemsDirs}
    print("Using menu items: %s" % menuItems)

    convertBBoxToDarknet()
    splitDataset()

