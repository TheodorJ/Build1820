"""
A simple Python script to send messages to a sever over Bluetooth using
Python sockets (with Python 3.3 or above).
"""

import socket

serverMACAddress = b'a4:cf:12:77:20:ca'
port = 3
# s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
# s.connect((serverMACAddress,i))
s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_SCO)
s.connect((serverMACAddress))

while 1:
    text = input()
    if text == "quit":
        break
    s.send(bytes(text, 'UTF-8'))
s.close()
