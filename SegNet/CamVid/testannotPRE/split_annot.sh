shopt -s nullglob
for file in ./*.tif
do
	slice-image "$file" 15
done
mv *.png ../testannotSpliced
shopt -u nullglob
