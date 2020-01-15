import board
import neopixel
pixels = neopixel.NeoPixel(board.D18,116, auto_write=False, pixel_order=neopixel.GRBW)
for i in range(0,len(pixels),4):
  pixels[i]=(100,0,0,0)
  pixels[i+1]=(0,100,0,0)
  pixels[i+2]=(0,0,100,0)
  pixels[i+3]=(0,00,0,100)
pixels.show()
