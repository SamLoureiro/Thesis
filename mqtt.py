import paho.mqtt.client as mqtt

# Define MQTT broker details
broker_address = "192.168.x.x"  # Replace with the local IP of your MQTT broker
broker_port = 1883  # Default MQTT port
mqtt_topic = "your_topic"  # The topic you want to subscribe to

# Callback function when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        client.subscribe(mqtt_topic)  # Subscribe to the topic after connection
    else:
        print(f"Connection failed with code {rc}")

# Callback function when a message is received from the broker
def on_message(client, userdata, message):
    print(f"Message received on topic {message.topic}: {message.payload.decode('utf-8')}")

# Create MQTT client instance
client = mqtt.Client()

# Assign callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to broker
client.connect(broker_address, broker_port)

# Start the loop to process network traffic and dispatch callbacks
client.loop_forever()


# 1833