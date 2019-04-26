
"""
BBox: [bounding box left X] [bounding box top Y] [bounding box right X] [bounding box bottom Y] [Label]
DN: [Label] [object center in X] [object center in Y] [object width in X] [object width in Y]
"""

class DarknetObject:
    def __init__(self, bBoxLine: str):
        bBoxPieces = bBoxLine.split(" ")
        label = bBoxPieces[4]
        x1, y1, x2, y2 = [int(i) for i in bBoxPieces[:4]]
        
        width = x2 - x1
        height = y2 - y1
        x = x1 + width / 2
        y = y1 + height / 2
    
        dnPieces = [str(i) for i in [label, x, y, width, height]]
        self.dnLine = " ".join(dnPieces)
        