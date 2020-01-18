docker run --rm --name local_mqtt_broker --network hw03_local -p 1883:1883  -v /tmp:/tmp -v $(pwd)/python:/local_mqtt_broker -ti local_mqtt_broker /bin/ash
