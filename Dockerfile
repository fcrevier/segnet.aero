FROM caffe_segnet:cpu

RUN pip install numpy --upgrade
ADD . /
