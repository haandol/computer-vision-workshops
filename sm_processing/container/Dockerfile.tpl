FROM $BASE_IMAGE

RUN apt-get update
RUN apt-get -y install libglib2.0 python-opencv
RUN python3 -m pip install opencv-python

ENV OPENCV_PREINSTALLED=1
