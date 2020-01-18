docker run -e DISPLAY=$DISPLAY --rm  --privileged -v /tmp:/tmp -v $(pwd)/python:/face_detect -ti face_detector
