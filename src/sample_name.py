
SAMPLE_NAME_FORMAT = "{:03d}/{:03d}-{:d}.{}"
DATASET_SAMPLE_NAME_FORMAT = "{:03d}-{:d}.{}"

class SampleName:

    def __init__(self, menuItemId: int, number: int):
        self.menuItemId = menuItemId
        self.number = number

    @classmethod
    def fromFilename(cls, filename: str):
        pieces: List[str] = filename.split("-")
        menuItemId: int = int(pieces[0])
        number: int = int(pieces[1].split(".")[0])
        return cls(menuItemId, number)

    def getFilename(self, extension) -> str:
        return SAMPLE_NAME_FORMAT.format(self.menuItemId, self.menuItemId, self.number, extension)
    
    def getDatasetFilename(self, extension) -> str:
        return DATASET_SAMPLE_NAME_FORMAT.format(self.menuItemId, self.number, extension)
