import paho.mqtt.client as mqtt
import time

# This is the Subscriber

def on_connect(client, userdata, flags, rc):
  print("Connected with result code "+str(rc))
  client.subscribe("topic/test")

def on_message(client, userdata, msg):
    filename = str(int(round(time.time() * 1000))) + '.png'
    print(filename)
    f = open('output/' + filename, 'w+b')
    f.write(msg.payload)
    f.close()

#  if msg.payload.decode() == "Hello world!":
#    print("Yes!")
#    client.disconnect()
    
client = mqtt.Client()
client.connect("remote_mqtt_broker",1883,60)
client.on_connect = on_connect
client.on_message = on_message

client.loop_forever()
