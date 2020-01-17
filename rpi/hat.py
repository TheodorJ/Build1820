import telnetlib
from binascii import hexlify
import time
import os

from shivi_pixel import endgame, draw_spell

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
game_state = "GAME_END"

player_health = {}
player_last_hor_defend = {}
player_last_ver_defend = {}
defend_duration = 1500

spell_in_air = False
spell_birthday = now()
spell_lifespan = 3000 # 3 seconds
spell_type = "ERR"
spell_sender = None

rebound = False

# IPs of the wands
ips = ["192.168.4.11","192.168.4.12"]
# telnet handles
tns = {}

def other_player(handle):
    return [x for x in player_health.keys() if x != handle][0]

def player_is_hor_defended(player):
    return (now() - player_last_hor_defend[player]) < defend_duration

def player_is_ver_defended(player):
    return (now() - player_last_ver_defend[player]) < defend_duration

def send_spell(spell, sender):
    global spell_in_air
    global spell_sender
    global spell_type
    global spell_birthday
    spell_in_air = True
    spell_sender = sender
    spell_type = spell
    spell_birthday = now()

def broadcast(msg):
    for ip in ips:
        tns[ip].write(bytes(msg, "UTF-8"))

    print("Broadcasting %s" % msg)

def ip_to_player(ip):
    if ip == ips[0]:
        return 0
    else:
        return 1

def display_cast_LEFT(ip):
    try:
        pid = os.fork()
        if pid > 0:
            # parent process, return and keep running
            return
    except:
        pass

    player = ip_to_player(ip)
    # Execute display
    draw_spell(player, 0)

    None.fake_error

def display_cast_DOWN(player):
    try:
        pid = os.fork()
        if pid > 0:
            # parent process, return and keep running
            return
    except:
        pass

    player = ip_to_player(ip)
    # Execute display
    draw_spell(player, 3)

    None.fake_error

def display_GAME_END():
    try:
        pid = os.fork()
        if pid > 0:
            # parent process, return and keep running
            return
    except:
        pass

    # Execute display
    endgame()

    exit(0)

def display_defend_LEFT(player):
    try:
        pid = os.fork()
        if pid > 0:
            # parent process, return and keep running
            return
    except:
        pass

    player = ip_to_player(ip)
    # Execute display
    draw_spell(player, 1)

    None.fake_error

def display_defend_DOWN(player):
    try:
        pid = os.fork()
        if pid > 0:
            # parent process, return and keep running
            return
    except:
        pass

    player = ip_to_player(ip)
    # Execute display
    draw_spell(player, 2)

    None.fake_error

def process_message(ip, value):
    """
    handle -- integer, characteristic read handle the data was received on
    value -- bytearray, the data returned in the notification
    """
    print("Received data: %s" % value)

    global game_state
    global num_wands_ready

    if ip not in player_health.keys():
        player_health[ip] = 2
        player_last_hor_defend[ip] = now()
        player_last_ver_defend[ip] = now()

    if(value == "V"): # button down
        if game_state == "GAME_END":
            num_wands_ready += 1
        if(num_wands_ready == 2):
            game_state = "GAME_START"
            broadcast("B")
            num_wands_ready = 0

    if(value == "^"): # button up
        if game_state == "GAME_END":
            num_wands_ready -= 1

    if(value in spells): # cast
        if(game_state == "GAME_START"):
            if(value == "L"):   # LEFT
                send_spell("LEFT", ip)
                display_cast_LEFT(ip)
            elif(value == "R"): # RIGHT
                player_last_hor_defend[ip] = now()
                display_defend_LEFT(ip)
            elif(value == "U"): # UP
                player_last_ver_defend[ip] = now()
                display_defend_DOWN(ip)
            elif(value == "D"): # DOWN
                send_spell("DOWN", ip)
                display_cast_DOWN(ip)
            else:             # Error
                pass


for ip in ips:
    tns[ip] = telnetlib.Telnet(ip)
    print("Connected to " + ip)

while 1:
    # Main game loop

    if spell_in_air and (now() - spell_birthday) > spell_lifespan:
        if spell_type == "LEFT":
            # If the other player isn't defended, subtract a health point
            other_p = other_player(spell_sender)

            if not player_is_hor_defended(other_p):
                print("Hit other player!")
                player_health[other_p] -= 1
            elif rebound:
                send_spell("LEFT", other_p)

            if player_health[other_p] == 0:
                # GAME END
                print("Ending game!")
                game_state = "GAME_END"
                broadcast("E")
                player_health[spell_sender] = 1
                player_last_hor_defend[spell_sender] = now()
                player_last_ver_defend[spell_sender] = now()
                player_health[other_p] = 1
                player_last_hor_defend[other_p] = now()
                player_last_ver_defend[other_p] = now()
                display_GAME_END()
                exit(0)

            spell_in_air = False
        if spell_type == "DOWN":
            # If the other player isn't defended, subtract a health point
            other_p = other_player(spell_sender)

            if not player_is_ver_defended(other_p):
                player_health[other_p] -= 1
            elif rebound:
                send_spell("DOWN", other_p)

            if player_health[other_p] == 0:
                # GAME END
                game_state = "GAME_END"
                broadcast("E")
                player_health[spell_sender] = 1
                player_last_hor_defend[spell_sender] = now()
                player_last_ver_defend[spell_sender] = now()
                player_health[other_p] = 1
                player_last_hor_defend[other_p] = now()
                player_last_ver_defend[other_p] = now()
                display_GAME_END()
                exit(0)

            spell_in_air = False


    for ip in ips:
        tn = tns[ip]
        try:
            spell = tn.read_eager().decode("UTF-8")

            # read_eager returns "" if no data available
            if (spell != ""):
                print("received " + spell + " on " + ip)

                # Process each letter in the message
                for c in spell:
                    process_message(ip, c)

        except EOFError:
            print(ip + " disconnected")
