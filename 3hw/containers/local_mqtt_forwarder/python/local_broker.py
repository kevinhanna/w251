import paho.mqtt.client as mqtt
import time

remote_client = mqtt.Client()
# This is the Subscriber

def post_face_remote(face):
    remote_client.connect("169.53.167.199",1883,60)
    remote_client.publish("topic/test", payload=face, qos=1, retain=False);
    remote_client.disconnect();

def on_connect(client, userdata, flags, rc):
  print("Connected with result code "+str(rc))
  client.subscribe("topic/test")

def on_message(client, userdata, msg):
    print("Forwarding face")
    post_face_remote(msg.payload)

    #filename = str(int(round(time.time() * 1000))) + '.png'
    #f = open('output/' + filename, 'w+b')
    #f.write(msg.payload)
    #f.close()

    
client = mqtt.Client()
client.connect("local_mqtt_broker",1883,60)
client.on_connect = on_connect
client.on_message = on_message

client.loop_forever()
