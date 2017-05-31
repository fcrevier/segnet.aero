# segnet.aero
CS231N - Project


## Use docker for workflow.
>$ sudo docker build -t cs231n:latest .
don't forget the '.' at the end of that command
## mount a local directory in the docker:
>$ sudo docker run -it -v /home/tobyb/ML/segnet.aero/testVolume/:/test dockImage
must be full path
and the -v mounts my local ./testVolume directory into the container at location /test (absolute path). Already tested  changing files on host and the changes persist in real-time to the container.
