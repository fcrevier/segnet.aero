#!/bin/bash
rm test.txt
a=$(pwd)

for file in ./test/*.png
do
	v1=$(realpath "$file")
	end=$(basename "$v1")
	v2=$a'/testannot/'$end
	echo $v1' '$v2 >> test.txt
done
