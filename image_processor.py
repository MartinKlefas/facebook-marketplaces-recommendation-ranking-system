import cv2
from tqdm import tqdm
import os

from torchvision import transforms

defaultSize = 224

def resize_image(final_size, im):
    size = im.shape[:2]
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

    
    
    total = len(dirs)
    new_images = dict()

    for item in tqdm(dirs,bar_format='{l_bar}{bar}| {percentage:3.0f}% {n}/{total} [{remaining}{postfix}]',desc="Resizing"):
        if os.path.isfile(path + item):
            im = cv2.imread(path + item)
            new_images[item]= resize_image(final_size, im)
    
    return new_images
            





def fullPreProcess_Image(filePath: str):
    image = processImage(thisPath= filePath)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform(image)
