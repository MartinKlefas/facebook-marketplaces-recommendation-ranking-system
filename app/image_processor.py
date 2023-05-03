import cv2
import os

from torchvision import transforms
import torch

defaultSize = 224

def resize_image(final_size, im):
    size = im.shape[:2]
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    new_im = cv2.resize(im,new_image_size,interpolation= cv2.INTER_LINEAR)
    return new_im

def processImage(thisPath : str = "" , thisImage :cv2.Mat = None, thisBuffer = None):
    if thisImage:
        return(resize_image(defaultSize,thisImage))
    
    if thisPath:
        myImage = cv2.imread(thisPath)
        os.unlink(thisPath)
        return(resize_image(defaultSize,myImage))
    
    if thisBuffer:
        myImage = cv2.imdecode(thisBuffer,cv2.IMREAD_COLOR)
        return(resize_image(defaultSize,myImage))
    

def fullPreProcess_Image(filePath: str):
    image = processImage(thisPath= filePath)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform(image)

    

def getEmbedding(image, model):
    device = torch.device("cpu")

    

    pyImage = fullPreProcess_Image(filePath=image).unsqueeze(0)

    layer = model._modules.get('avgpool')

    my_embedding = torch.zeros(2048)

    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))

    h = layer.register_forward_hook(copy_data)

    model(pyImage.to(device))

    h.remove()

    return my_embedding
