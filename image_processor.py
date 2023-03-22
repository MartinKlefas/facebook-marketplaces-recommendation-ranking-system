import cv2
from tqdm import tqdm
import os, sys, argparse

defaultSize = 224

argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--image", help="An Image File")
argParser.add_argument("-o", "--out", help="Output Folder Name or File Name")
argParser.add_argument("-d", "--dir", help="A Directory of images")
argParser.add_argument("-s", "--size", help="Maximum pixel size in any dimension")

args = argParser.parse_args()

if args.size:
    defaultSize = args.size

def resize_image(final_size, im):
    size = im.shape
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    new_im = cv2.resize(im,new_image_size,interpolation= cv2.INTER_LINEAR)
    return new_im

def validate_output_dir(path):
    try:
    # Check if the folder exists or not
        if not os.path.isdir(path):
            
            # If not then make the new folder
            os.mkdir(path)

        return True, None
    except Exception as ex:

        return False, ex

def processImage(thisPath : str = "" , thisImage :cv2.Mat = None):
    if thisImage:
        return(resize_image(defaultSize,thisImage))
    
    if thisPath:
        myImage = cv2.imread(thisPath)
        return(resize_image(defaultSize,myImage))
    
def processFolder(path: str, size: int = defaultSize):
    if not path[:-1] =="/":
        path = path + "/"

    dirs = os.listdir(path)
    final_size = size

    # Check if the output folder exists and create it if possible
    Created, ErrorCode = validate_output_dir(path + "clean_image_data/")
    if not Created : # then there was an error
        print(ErrorCode)
        raise ErrorCode
    
    total = len(dirs)

    for item in tqdm(dirs,bar_format='{l_bar}{bar}| {percentage:3.0f}% {n}/{total} [{remaining}{postfix}]',desc="Resizing"):
        if os.path.isfile(path + item):
            im = cv2.imread(path + item)
            new_im = resize_image(final_size, im)
            new_im.save(f'{path}clean_image_data/{item}')



if args.image :
    try:
        myImage = cv2.imread(args.image)
        myImage = processImage(thisimage= myImage)
        if args.out :
            cv2.imwrite(filename= args.out,image= myImage)
        else:
            cv2.imwrite(filename= args.image, image= myImage)

    except Exception as ex:
        print(ex)

if args.dir :
    try:
        processFolder(args.dir)
    except Exception as ex:
        print(ex)
