import board
import neopixel
import time

pixels = neopixel.NeoPixel(board.D18,114, auto_write=False, pixel_order=neopixel.GRBW)

def draw_spell(player, spell):
  color = (spell == 0)? (100, 0, 0, 0) : (spell == 1)? (0, 100, 0, 0) :
          (spell == 2)? (0, 0, 100, 0) : (0, 0, 0, 100)

  for(int i = 0; i < len(pixels); i++):
    if(player == 0):
      if(i < 56):
        pixels[i] = color
      else:
        pixels[114-1-(i-56)] = color
    else:
      if( i < 58):
        pixels[i+56] = color
      else:
        pixels[56 - (58-i)-1] = color
    
    pixels.show()
    time.sleep(0.5)



while True:
  play = input("Enter player 0/1:")
  spl  = input("Enter spell 0/1/2/3")

  draw_spell(play, spl)
for i in range(0,len(pixels),4):
  pixels[i]=(100,0,0,0)
  pixels[i+1]=(0,100,0,0)
  pixels[i+2]=(0,0,100,0)
  pixels[i+3]=(0,00,0,100)
pixels.show()
