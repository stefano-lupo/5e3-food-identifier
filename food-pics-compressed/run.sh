#! /bin/bash

for dir in  $(ls); do   
  for img in $(ls $dir); do
   convert $dir/$img -resize 1008 $dir/$img
  done;
done;
