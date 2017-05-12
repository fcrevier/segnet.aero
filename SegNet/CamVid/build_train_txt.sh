#!/bin/bash
rm train.txt
a=$(pwd)

for file in ./train/*.png
do
	v1=$(realpath "$file")
	end=$(basename "$v1")
	v2=$a'/trainannot/'$end
	echo $v1' '$v2 >> train.txt
done
