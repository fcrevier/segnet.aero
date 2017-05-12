shopt -s nullglob
for file in ./*.tif
do
	slice-image -d ../trainannot "$file" 15
done
shopt -u nullglob