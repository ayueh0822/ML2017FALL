import sys
from PIL import Image

def main(argv):
    img_input = Image.open(argv[0])
    img_output = Image.new(img_input.mode, img_input.size )
    pixels_input = img_input.load()
    pixels_output = img_output.load()

    for i in range(img_output.width):    
        for j in range(img_output.height):
            #print (pixels_input[i,j])
            pixels_output[i,j] = ( int(pixels_input[i,j][0]/2), int(pixels_input[i,j][1]/2), int(pixels_input[i,j][2]/2) )

    #img_output.show()
    img_output.save('Q2.png’)
if __name__ == '__main__':
    main(sys.argv[1:])
