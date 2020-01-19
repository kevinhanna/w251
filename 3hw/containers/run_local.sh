docker network create --driver bridge hw03_local
# Start up local broker
docker run --rm --name local_mqtt_broker --network hw03_local --detach -p 1883:1883 local_mqtt_broker mosquitto

# Start up face thingy

