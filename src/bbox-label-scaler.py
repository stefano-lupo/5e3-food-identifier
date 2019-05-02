import os

base_dir = "../labels"
new_base_dir = "../labels-scaled"

SCALE = 0.5
formatStr = "{} {} {} {} {}\n"

for menuDir in os.listdir(base_dir):
    fullDir = os.path.join(base_dir, menuDir)
    for labelFile in os.listdir(fullDir):
        with open(os.path.join(fullDir, labelFile), 'r') as f:
            bBoxLines = [line.strip("\n") for line in f.readlines()]
            with open(os.path.join(new_base_dir, menuDir, labelFile), 'w') as f2:
                f2.write(bBoxLines[0] +  "\n")
                for line in bBoxLines[1:]:
                    line = line.split(" ")
                    x1, y1, x2, y2 = [round(int(l)*SCALE) for l in line[0:4]]
                    label = " ".join(line[4:])
                    f2.write(formatStr.format(x1, y1, x2, y2, label))

