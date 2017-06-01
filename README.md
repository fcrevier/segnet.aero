# segnet.aero
CS231N - Project


## Use docker for workflow. (change gpu to cpu for the cpu version)
download my docker of caffe_segnet:
`sudo docker pull buckleytoby/caffe_segnet:gpu`
get docker image id:
`sudo docker images`
change name:
`sudo docker tag [imageID] caffe_segnet:gpu`
`sudo docker build -t cs231n:latest .`
don't forget the '.' at the end of that command
## mount a local directory in the docker:
`sudo docker run -it -v $(pwd)/SegNet/:/SegNet cs231n:gpu`
must be full path
and the -v mounts my local ./testVolume directory into the container at location /test (absolute path). Already tested  changing files on host and the changes persist in real-time to the container.
