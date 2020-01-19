docker run --rm --name remote_mqtt_broker  --network hw03_remote -p 1883:1883  -v /tmp:/tmp -v $(pwd)/python:/remote_mqtt_broker -ti remote_mqtt_broker /bin/ash
