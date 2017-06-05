
setting up the dataset steps:
1. wget the index.html from Mnih's road dataset webpage https://www.cs.toronto.edu/~vmnih/data/
2. using vi macros, strip off html tags from each image link
3. run >>wget -i index.html
	this will download each line from index.html (each image) to the current folder
4. modify split_images.sh to point to correct directories, then run it
	- must include anaconda libraries
4b. if number of files (ls train | wc -l) doesn't match between train and annot, do >>diff <(ls train) <(ls trainannot)
5. fix annotated images so that the pixel value is from 1 to 0 (for our 2 classes)
	>>cd trainannot
	>>python ../fix_annot.py
6a. delete 11878765_15_02_04.png in train and trainannot because it doesn't work for some reason.
6b. build the SegNet input file by running build_[train,test]_txt.sh from CamVid dir
7. change the model parameters in order to fit the input image size

 -- Toby
