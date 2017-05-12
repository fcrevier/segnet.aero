#!/bin/bash
rm train/*.png
cd trainPRE

shopt -s nullglob
for file in ./*.tiff
do
	slice-image -d ../train "$file" 15
done
shopt -u nullglob

cd ..
rm trainannot/*.png
cd trainannotPRE


shopt -s nullglob
for file in ./*.tif
do
	slice-image -d ../trainannot "$file" 15
done
shopt -u nullglob

