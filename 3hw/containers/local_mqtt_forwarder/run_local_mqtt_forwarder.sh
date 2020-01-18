docker run --rm --name local_mqtt_forwarder --network hw03_local -v /tmp:/tmp -v $(pwd)/python:/local_mqtt_forwarder -ti local_mqtt_forwarder /bin/ash
