docker run --rm --name remote_mqtt_processor --privileged=true --network hw03_remote -v /tmp:/tmp -v $(pwd)/python:/remote_mqtt_processor -ti remote_mqtt_processor /bin/ash 
