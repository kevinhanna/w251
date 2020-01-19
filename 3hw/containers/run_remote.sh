docker network create --driver bridge hw03_remote
# Start up remote broker
docker run --rm --name remote_mqtt_broker --network hw03_remote --detach -p 1883:1883 remote_mqtt_broker mosquitto

