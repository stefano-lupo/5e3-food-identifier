import json, os, cv2

from typing import List, Dict, Set

EXTRA_SCALE = 2

DEFAULT_PLOTLY_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207)
]

class Result:
    def __init__(self, result):
        self.filename: str = result['filename']
        self.objects: List[Object] = [Object(o) for o in result['objects']]

class Object:
    def __init__(self, obj):
        self.classId: int = obj['class_id']
        self.name: str = obj['name']
        coords =  obj['relative_coordinates']
        self.xCenter: float = coords['center_x']
        self.yCenter: float = coords['center_y']
        self.width: float = coords['width']
        self.height: float = coords['height']
        self.confidence = obj['confidence']

    def centerToCorners(self, scale=(1, 1)) -> List[int]:
        halfWidth = self.width / 2
        halfHeight = self.height / 2

        scale = (scale[0] * EXTRA_SCALE, scale[1] * EXTRA_SCALE)

        x1 = self.xCenter - halfWidth
        y1 = self.yCenter - halfHeight
        x2 = self.xCenter + halfWidth
        y2 = self.yCenter + halfHeight

        coords = [x1 * scale[0], y1 * scale[1], x2 * scale[0], y2 * scale[1]]
        return [round(c) for c in coords]

    def draw(self, img):
        height, width, _ = img.shape
        x1, y1, x2, y2 = self.centerToCorners(scale=(width, height))
        cv2.rectangle(img, (x1, y1), (x2, y2), DEFAULT_PLOTLY_COLORS[self.classId], 1)


RESULTS_FILE = "result.json"


def loadResults() -> List[Result]:
    with open(RESULTS_FILE, 'r') as f:
        resultsJson = json.load(f)
        results: List[Result] = [Result(r) for r in resultsJson]
        print(results)

    return results

def drawBoundingBoxes(result: Result):
    img = cv2.imread(result.filename, cv2.IMREAD_COLOR)

    # img = cv2.flip(img, -1)
    for obj in result.objects:
        obj.draw(img)

    cv2.imshow('image', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    results: List[Result] = loadResults()
    for result in results:
        drawBoundingBoxes(result)
