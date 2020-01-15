#!/usr/bin/env python3
import binascii
import pygatt
import time

# Useful examples:
# https://github.com/pcborenstein/bluezDoc/wiki/hcitool-and-gatttool-example
# https://unix.stackexchange.com/questions/288978/how-to-configure-a-connection-interval-in-a-ble-connection


DEVICE_ADDR = "a4:cf:12:77:20:ca"
TEST_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
MAX_MSG_LEN = 300

# for uuid in device.discover_characteristics().keys():
#     print("UUID %s:" % uuid)

adapter = pygatt.GATTToolBackend()

try:
    t0 = time.monotonic()
    adapter.start()
    device = adapter.connect(DEVICE_ADDR)
    device.exchange_mtu(MAX_MSG_LEN + 4)
    t1 = time.monotonic()
    print("Setup time: " + str(t1 - t0))

    input("ready?")
    t0 = time.monotonic()
    value = device.char_read(TEST_UUID)
    t1 = time.monotonic()
    print("Read time: " + str(t1 - t0))
    print(value, len(value))


    t0 = time.monotonic()
    device.char_write(TEST_UUID, bytearray("beep boop really long new value lkuyasdfoniausnodfuyasnoiudyvasoiduyfoa sudfh oasudhf oasdfh iasudhf oashd ofahusdf oashdof ash dfoaushdoashofash oduas idufh aosuidfh oasiuudfh oasiudfh oasuidhf oasduih foausidhf oaiusuhdf oiausdh foiuahsd", "UTF-8"))
    t1 = time.monotonic()
    print("Write time: " + str(t1 - t0))

    t0 = time.monotonic()
    value = device.char_read(TEST_UUID)
    t1 = time.monotonic()
    print("Read time: " + str(t1 - t0))
    print(value, len(value))

    t0 = time.monotonic()
    device.char_write(TEST_UUID, bytearray("no", "UTF-8"))
    t1 = time.monotonic()
    print("Write time: " + str(t1 - t0))

    t0 = time.monotonic()
    for i in range(100):
        value = device.char_read(TEST_UUID)
    t1 = time.monotonic()
    print("Read time: " + str(t1 - t0))
    print(value, len(value))
finally:
    adapter.stop()
