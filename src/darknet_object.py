import cv2

"""
BBox: [bounding box left X] [bounding box top Y] [bounding box right X] [bounding box bottom Y] [Label]
DN: [Label] [object center in X] [object center in Y] [object width in X] [object width in Y]
"""
INGREDIENTS = [
    "Fork",
    "Knife",
    "Spoon",
    "Glass",
    "Rice",
    "Chicken",
    "Sweetcorn",
    "Noodle",
    "Brocolli",
    "Tomato",
    "Red Pasta",
    "Chorizo",
    "Soup",
    "Brown Bread",
    "Yellow Pasta",
    "Rocket",
    "Apple",
    "Grape",
    "White Pasta"
]

INGREDIENT_IDS = {}
for i, ing in enumerate(INGREDIENTS):
    INGREDIENT_IDS[ing] = i


IMG_WIDTH = 1008
IMG_HEIGHT = 756


class DarknetObject:
    def __init__(self, bBoxLine: str, normalize: bool = False):
        bBoxPieces = bBoxLine.split(" ")
        self.label = INGREDIENT_IDS[" ".join(bBoxPieces[4:])]
        x1, y1, x2, y2 = [int(i) for i in bBoxPieces[:4]]

        self.topLeft = (x1, y1)
        self.bottomRight = (x2, y2)

        width = x2 - x1
        height = y2 - y1
        x = round(x1 + width / 2)
        y = round(y1 + height / 2)

        if normalize:
            width = width / IMG_WIDTH
            height = height / IMG_HEIGHT
            x = x / IMG_WIDTH
            y = y / IMG_HEIGHT
    
        dnPieces = [str(i) for i in [self.label, x, y, width, height]]
        self.dnLine = " ".join(dnPieces)

    def drawBoundingBox(self, img):
        cv2.rectangle(img, self.topLeft, self.bottomRight, (0, 255, 0), 3)
