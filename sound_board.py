import RPi.GPIO as GPIO
import time
import threading
from enum import Enum

class SoundPin(Enum):
    WIZARD_ONE_WIN = 1
    WIZARD_TWO_WIN = 2
    HEDWIG         = 3
    YOU_SUCK       = 4
    OH_NO          = 5
    EXPLOSION      = 6
    READY_BEGIN    = 7
    FIRE           = 8
    EXPLOSION2     = 9
    MAGIC          = 10


GPIO.setmode(GPIO.BCM)

for pin in range(1, 11):
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)

def play_sound_pin(on_pin):
    for pin in range(1, 11):
        if pin == on_pin:
            GPIO.output(pin, GPIO.LOW)
        else:
            GPIO.output(pin, GPIO.HIGH)

    threading.Timer(0.2, lambda: GPIO.output(on_pin, GPIO.HIGH)).start()


while True:
    on_pin = int(input("Enter pin:"))
    play_sound_pin(on_pin)



