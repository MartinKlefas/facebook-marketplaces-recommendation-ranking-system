from PIL import Image
from tqdm import tqdm
import os, sys

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.LANCZOS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
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

def clean_image_data(path: str, size: int):
    if not path[:-1] =="/":
        path = path + "/"

    dirs = os.listdir(path)
    final_size = size

    # Check if the output folder exists and create it if possible
    Created, ErrorCode = validate_output_dir(path + "clean_image_data/")
    if not Created : # then there was an error
        print(ErrorCode)
        sys.exit()
    
    total = len(dirs)

    for item in tqdm(dirs,bar_format='{l_bar}{bar}| {percentage:3.0f}% {n}/{total} [{remaining}{postfix}]',desc="Resizing"):
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            new_im = resize_image(final_size, im)
            new_im.save(f'{path}clean_image_data/{item}.jpg')
