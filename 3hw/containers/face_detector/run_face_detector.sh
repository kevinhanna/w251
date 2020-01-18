docker run --name face_detector -e DISPLAY=$DISPLAY --rm  --privileged --network hw03_local -v /tmp:/tmp -v $(pwd)/python:/face_detect -ti face_detector
