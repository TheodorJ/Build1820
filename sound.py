# for other volume
# amixer set PCM -- 0 

import pygame

def play_sound(filename):
    try:
        pygame.init()
        pygame.mixer.Sound(filename).play()
    except:
        print("Error playing sound", filename)



if __name__ == "__main__":
    pygame.init()

    sound = pygame.mixer.Sound("sound/you_suck.wav")
    sound.play()
    input()  # for example code
