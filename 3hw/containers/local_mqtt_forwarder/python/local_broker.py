import paho.mqtt.client as mqtt
import time

remote_client = mqtt.Client()
# This is the Subscriber

def post_face_remote(face):
    remote_client.connect("169.53.167.199",1883,20)
    try:
        remote_client.publish("topic/test", payload=face, qos=0, retain=False);
    except:
        print("Unexpected error:", sys.exc_info()[0])
    remote_client.disconnect();

def on_connect(client, userdata, flags, rc):
  print("Connected with result code "+str(rc))
  client.subscribe("topic/test")

def on_message(client, userdata, msg):
#    if (len(msg.payload) < 50000):
    print("Forwarding face " + str(len(msg.payload)))
    post_face_remote(msg.payload)
#    else:
#        print("NOT forwarding face " + str(len(msg.payload)))

client = mqtt.Client(transport="tcp")

client.connect("local_mqtt_broker",1883,60)
client.on_connect = on_connect
client.on_message = on_message

client.loop_forever()
