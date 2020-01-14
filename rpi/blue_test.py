#!/usr/bin/env python
import binascii
import pygatt

DEVICE_ADDR = "a4:cf:12:77:20:ca"
TEST_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
MAX_MSG_LEN = 32

# for uuid in device.discover_characteristics().keys():
#     print("UUID %s:" % uuid)

adapter = pygatt.GATTToolBackend()

try:
    adapter.start()
    device = adapter.connect(DEVICE_ADDR)
    device.exchange_mtu(MAX_MSG_LEN + 4)
    value = device.char_read(TEST_UUID)
    print(value, len(value))
finally:
    adapter.stop()
