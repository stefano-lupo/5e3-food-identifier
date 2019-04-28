import os, shutil
from typing import List, Dict

from darknet_object import DarknetObject
from sample_name import SampleName

class DarknetSample:

    def __init__(self, imageDir: str, labelDir: str, sampleName: SampleName, normalize: bool = True):
        self.sampleName = sampleName
        self.imageDir = imageDir
        self.labelDir = labelDir
        with open(os.path.join(labelDir, sampleName.getFilename("txt"))) as f:
            # first line is junk in BBox Multiclass
            bBoxLines = [line.strip("\n") for line in f.readlines()[1:]]
            self.objects: List[DarknetObject] = [DarknetObject(line, normalize) for line in bBoxLines]

    def getLines(self) -> str:
        return "\n".join([obj.dnLine for obj in self.objects])
    
    def getImagePath(self) -> str:
        return os.path.join(self.imageDir, self.sampleName.getImageFilename())

    def writeSample(self, directory: str):
        labelFile: str = os.path.join(directory, self.sampleName.getDatasetFilename("txt"))
        with open(labelFile, 'w') as f:
            f.writelines(self.getLines())
        
        imgSrc: str = os.path.join(self.imageDir, self.sampleName.getFilename("jpg"))
        imgDest: str = os.path.join(directory, self.sampleName.getDatasetFilename("jpg"))
        shutil.copy(imgSrc, imgDest)

