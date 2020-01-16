import pygatt
from binascii import hexlify

import time
def now():
    return int(round(time.time() * 1000))

adapter = pygatt.GATTToolBackend()
addr = "a4:cf:12:77:11:1a" # The one not in the wand
# addr = "a4:cf:12:77:20:ca" # The one in the wand

num_wands_ready = 0
state = "GAME_END"

player_health = {}
player_last_hor_defend = {}
player_last_ver_defend = {}
defend_duration = 1000

def other_player(handle):
    return [x for x in player_health.keys() if x != handle][0]

def player_is_hor_defended(player):
    return (now() - player_last_hor_defend[player]) > defend_duration

def player_is_ver_defended(player):
    return (now() - player_last_ver_defend[player]) > defend_duration


def handle_data(handle, value):
    """
    handle -- integer, characteristic read handle the data was received on
    value -- bytearray, the data returned in the notification
    """
    print("Received data: %s (hex %s)" % (value, hexlify(value)))

    if handle not in player_health.keys():
        player_health[handle] = 5
        player_last_hor_defend[handle] = now()
        player_last_ver_defend[handle] = now()

    if(value == "BTN_DOWN"):
        if state == "GAME_END":
            num_wands_ready += 1
        if(num_wands_ready == 2):
            state = "GAME_START"

    if(value == "BTN_UP"):
        if state == "GAME_END":
            num_wands_ready -= 1

    if(value[:6] == "CAST "):
        if(state == "GAME_START"):
            spell = value[6]
            if(spell == 0):   # LEFT
                # If the other player isn't defended, subtract a health point
                other_p = other_player(handle)

                if not player_is_hor_defended(other_p):
                    player_health[other_p] -= 1

                if player_health[other_p] == 0:
                    # GAME END
                    state = "GAME_END"

            elif(spell == 1): # RIGHT
                player_last_hor_defend[handle] = now()
            elif(spell == 2): # UP
                player_last_ver_defend[handle] = now()
            elif(spell == 3): # DOWN
                # If the other player isn't defended, subtract a health point
                other_p = other_player(handle)

                if not player_is_ver_defended(other_p):
                    player_health[other_p] -= 1

                if player_health[other_p] == 0:
                    # GAME END
                    state = "GAME_END"
            else:             # Error
                pass


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
