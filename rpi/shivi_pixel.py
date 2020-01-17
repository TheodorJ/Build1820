import board
import neopixel
import time

pixels = neopixel.NeoPixel(board.D18,114, auto_write=False, pixel_order=neopixel.GRBW)

def draw_spell(player, spell):
  color = 0
  speed = 0.016
  if (spell == 0):        #left
    color = (50, 0, 0, 0)
  elif (spell == 1):      #right
    color = (0, 0, 0, 50)
    speed /= 6
  elif (spell == 2):      #up
    color = (0, 0, 50, 0)
    speed /= 6
  else:                   #down
    color =(0, 50, 0, 0)

  for  i in range(len(pixels)):
    if(player == 0):
      if(i < 56):
        pixels[i] = color
      else:
        pixels[114-1-(i-56)] = color
    else:
      if( i < 58):
        pixels[i+56] = color
      else:
        pixels[56 - (i-58)-1] = color

    pixels.show()
    time.sleep(speed)

  for  i in range(len(pixels)):
      pixels[i] = (0, 0, 0, 0)
  pixels.show()

def endgame():
  pixels.fill((0,0,0,0))
  pixels.show()
  time.sleep(0.5)
  pixels.fill((100,100,0,0))
  pixels.show()
  time.sleep(0.25)
  pixels.fill((0,0,0,0))
  pixels.show()
  time.sleep(0.25)
  pixels.fill((100,100,0,0))
  pixels.show()
  time.sleep(0.25)
  pixels.fill((0,0,0,0))
  pixels.show()
  time.sleep(0.25)
  pixels.fill((100,100,0,0))
  pixels.show()
  time.sleep(0.25)
  pixels.fill((0,0,0,0))
  pixels.show()
  time.sleep(0.25)
  pixels.fill((100,100,0,0))
  pixels.show()
  time.sleep(0.25)
  pixels.fill((0,0,0,0))
  pixels.show()
  time.sleep(0.25)

if __name__ == "__main__":
    while True:
      play = int(input("Enter player 0/1: "))
      spl  = int(input("Enter spell 0/1/2/3: "))

      draw_spell(play, spl)
      pixels.fill((0, 0, 0, 0))
      pixels.show()
