import paho.mqtt.client as mqtt
import time

# This is the Subscriber

def on_connect(client, userdata, flags, rc):
  print("Connected with result code "+str(rc))
  client.subscribe("w251/hw03/faces", qos=1)

def on_message(client, userdata, msg):
    filename = str(int(round(time.time() * 1000))) + '.png'
    print(str(len(msg.payload)))
    try:
        f = open('/mnt/kh-object/hw03/faces/' + filename, 'w+b')
        f.write(msg.payload)
        f.close()
    except:
        print("Unexpected error:", sys.exc_info()[0])

client = mqtt.Client()
client.connect("remote_mqtt_broker",1883,20)
client.on_connect = on_connect
client.on_message = on_message

client.loop_forever()
