import paho.mqtt.client as mqtt

# This is the Publisher

client = mqtt.Client()
client.connect("local_mqtt_broker",1883,60)
client.publish("topic/test", "Hello world!");
client.disconnect();
