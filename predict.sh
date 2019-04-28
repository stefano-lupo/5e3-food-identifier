#! /bin/bash

evalDir="./training-files/dataset/eval"
predictionFilename="predictions.jpg"
model="yolov3-tiny"
weights=$1

mkdir predictions

for filename in  $evalDir/*.jpg; do
    darknet detector test training-files/darknet-configs/obj.data training-files/darknet-configs/$model.cfg $weights $filename
    mv predictionFilename predictions/$filename
done