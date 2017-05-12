
setting up the dataset steps:
1. wget the index.html from Mnih's road dataset webpage https://www.cs.toronto.edu/~vmnih/data/
2. using vi macros, strip off html tags from each image link
3. run >>wget -i index.html
	this will download each line from index.html (each image) to the current folder
4. modify split_images.sh to point to correct directories, then run it
5. fix annotated images so that the pixel value is from 1 to 0 (for our 2 classes)
6. build the SegNet input file by running build_[train,test]_txt.sh

 -- Toby
