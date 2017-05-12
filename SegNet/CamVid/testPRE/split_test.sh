shopt -s nullglob
for file in ./*.tiff
do
	slice-image "$file" 15
done
shopt -u nullglob

mv *.png ../test
