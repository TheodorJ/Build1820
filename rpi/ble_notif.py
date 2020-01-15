import pygatt
from binascii import hexlify

adapter = pygatt.GATTToolBackend()
addr = "a4:cf:12:77:11:1a" # The one not in the wand
# addr = "a4:cf:12:77:20:ca" # The one in the wand

def handle_data(handle, value):
    """
    handle -- integer, characteristic read handle the data was received on
    value -- bytearray, the data returned in the notification
    """
    print("Received data: %s (hex %s)" % (value, hexlify(value)))

try:
    adapter.start()
    device = adapter.connect(addr) #"a4:cf:12:77:20:ca")
    print("connected")

    device.subscribe("beb5483e-36e1-4688-b7f5-ea07361b26a8",
                     callback=handle_data)
    while(1):
        pass
finally:
    adapter.stop()
