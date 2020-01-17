import telnetlib
from binascii import hexlify
import time

def now():
    return int(round(time.time() * 1000))

"""
Network commands
all of which are single ASCII charactes

U/D/L/R - wand direction
V/^     - wand button down/up
B/E     - game begin/end

"""

spells = ["U", "D", "L", "R"]

num_wands_ready = 0
state = "GAME_END"

player_health = {}
player_last_hor_defend = {}
player_last_ver_defend = {}
defend_duration = 1000

# IPs of the wands
ips = ["192.168.4.11","192.168.4.12"]
# telnet handles
tns = {}

def other_player(handle):
    return [x for x in player_health.keys() if x != handle][0]

def player_is_hor_defended(player):
    return (now() - player_last_hor_defend[player]) > defend_duration

def player_is_ver_defended(player):
    return (now() - player_last_ver_defend[player]) > defend_duration


def process_message(ip, value):
    """
    handle -- integer, characteristic read handle the data was received on
    value -- bytearray, the data returned in the notification
    """
    print("Received data: %s (hex %s)" % (value, hexlify(value)))

    if handle not in player_health.keys():
        player_health[handle] = 5
        player_last_hor_defend[handle] = now()
        player_last_ver_defend[handle] = now()

    if(value == "V"): # button down
        if state == "GAME_END":
            num_wands_ready += 1
        if(num_wands_ready == 2):
            state = "GAME_START"

    if(value == "^"): # button up
        if state == "GAME_END":
            num_wands_ready -= 1

    if(value in spells): # cast
        if(state == "GAME_START"):
            if(value == "L"):   # LEFT
                # If the other player isn't defended, subtract a health point
                other_p = other_player(handle)

                if not player_is_hor_defended(other_p):
                    player_health[other_p] -= 1

                if player_health[other_p] == 0:
                    # GAME END
                    state = "GAME_END"

            elif(value == "R"): # RIGHT
                player_last_hor_defend[handle] = now()
            elif(value == "U"): # UP
                player_last_ver_defend[handle] = now()
            elif(value == "D"): # DOWN
                # If the other player isn't defended, subtract a health point
                other_p = other_player(handle)

                if not player_is_ver_defended(other_p):
                    player_health[other_p] -= 1

                if player_health[other_p] == 0:
                    # GAME END
                    state = "GAME_END"
            else:             # Error
                pass


for ip in ips:
    tns[ip] = telnetlib.Telnet(ip)
    print("Connected to " + ip)

while 1:
    # Main game loop

    for ip in ips:
        tn = tns[ip]
        try:
            spell = tn.read_eager()

            # read_eager returns "" if no data available
            if (spell != ""):
                print("received " + spell + " on " + ip)

                # Process each letter in the message
                for c in spell.decode("UTF-8"):
                    process_message(ip, c)

        except EOFError:
            print(ip + " disconnected")
