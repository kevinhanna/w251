import paho.mqtt.client as mqtt
import time

remote_client = mqtt.Client()
# This is the Subscriber

remote_client.connect("169.53.167.199",1883,60)
def post_face_remote(face):
    #remote_client.connect("169.53.167.199",1883,60)
    #remote_client.connect("169.53.167.193",1883,60)
    try:
        remote_client.publish("topic/test", payload=face, qos=0, retain=False);
    except:
        print("Unexpected error:", sys.exc_info()[0])
    #remote_client.disconnect();

def on_connect(client, userdata, flags, rc):
  print("Connected with result code "+str(rc))
  client.subscribe("topic/test",qos=1)

def on_message(client, userdata, msg):
    print("Forwarding face " + str(len(msg.payload)))
    post_face_remote(msg.payload)

client = mqtt.Client(transport="tcp")

client.connect("local_mqtt_broker",1883,60)
client.on_connect = on_connect
client.on_message = on_message

client.loop_forever()
