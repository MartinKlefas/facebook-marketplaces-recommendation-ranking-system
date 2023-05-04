import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch

import faiss, os, shutil, uuid
import numpy as np
from datetime import datetime
import pandas as pd


import image_processor

start_time = datetime.now()
def init(doModel : bool = False, doDevice : bool = False,doIndex : bool = False,doImageList : bool = False):
    if doModel or doDevice: 
        try:
            
            device = torch.device("cpu")


            model = torch.load('best_weights.pt',map_location=device)
            
            pass
        except:
            raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")
    else:
        model = None
        device = None

    if doIndex:
        try:
            index = faiss.read_index('images_faiss.index')
            pass
        except:
            raise OSError("No FAISS index found. Check that you have the encoder and the model in the correct location")
    else:
        index = None

    if doImageList:
        try:
            imageList =   pd.read_csv(filepath_or_buffer='training_data.csv')
            pass
        except:
            raise OSError("No image key dictionary found. Check that it's in the correct folder")
    else:
        imageList = None
        
    
    return model, device, index, imageList



app = FastAPI(
    title="FAISS Image Similarity API",
    description="This API accepts images for a similarity search based on a sample of images scraped from Facebook Marketplace. The intention being to see if a hypothetical user would be interested in a new item.",
    version="1.0.1",
    contact={
        "name": "Martin Klefas-Stennett",
        "url": "https://martinklefas.github.io/",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

tags_metadata = [
    {
        "name": "Health Check",
        "description": "Operations to check if the API is running and properly configured.",
    },
    {
        "name": "Predict",
        "description": "Operations which accept an input image and pass it through the trained model",
        
    },
]


@app.get('/healthcheck',tags=["Health Check"])
def healthcheck():
  timediff = datetime.now() - start_time


  msg = "API has been running for " + str(timediff )
  
  return {"message": msg}

@app.post('/healthcheck/file',tags=["Health Check"])
def healthcheck_file(image: UploadFile = File(...)):
  newPath = os.path.join(str(uuid.uuid4())+'_'+image.filename)
  try:
      with open(newPath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
      os.unlink(newPath)  
      return {"message": newPath}
  except Exception as ex:
      return {"message" : ex}
  

  
@app.post('/predict/feature_embedding',tags=["Predict"])
def Image_Embedding_String(image: UploadFile = File(...)):
    model, device, index, imageList = init(doModel=True)
    
    newPath = os.path.join(str(uuid.uuid4())+'_'+image.filename)#'received_files',str(uuid.uuid4())+'_'+image.filename)
    with open(newPath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    embedding = image_processor.getEmbedding(image=newPath,model=model)

    return JSONResponse(content={
    "features": embedding.tolist(), 
   
        })
  
@app.post('/predict/similar_image_indices',tags=["Predict"])
def show_similar_image_indeces(matches: str = 3,image: UploadFile = File(...)):
    model, device, index, imageList = init(doModel=True,doIndex=True)

    newPath = os.path.join(str(uuid.uuid4())+'_'+image.filename)#'received_files',str(uuid.uuid4())+'_'+image.filename)
    with open(newPath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    searchEmbedding = image_processor.getEmbedding(newPath,model=model)
    searchEmbedding = np.array(searchEmbedding,dtype="float32",ndmin=2)
    # if we look at https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/ 
    # we can actually have a "y" dimensional list of searchEmbeddings, and find matches for all of them with a single call.
    
    
    nprobe = 5
    distances, indices = index.search(x=searchEmbedding,k=matches)

    return JSONResponse(content={
    "similar_indices": indices.tolist(), # Return the index of similar images here
        })

@app.post('/predict/similar_images',tags=["Predict"])
def return_matching_image_filenames(matches: str = 3,image: UploadFile = File(...)):
    model, device, index, imageList = init(doModel=True,doImageList=True,doIndex=True)

    newPath = os.path.join(str(uuid.uuid4())+'_'+image.filename)
    with open(newPath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    searchEmbedding = image_processor.getEmbedding(newPath,model=model)
    searchEmbedding = np.array(searchEmbedding,dtype="float32",ndmin=2)

    nprobe = 5
    distances, indices = index.search(x=searchEmbedding,k=matches)

    image_ids = list()

    for array in indices:
        for idx in array:
            image_ids.append(imageList.at[idx,"id"])

    print(image_ids)

    return JSONResponse(content={
    "similar_images": image_ids, # Return the names of similar images here
        })
    
    
if __name__ == '__main__':
 
  uvicorn.run("api:app", host="0.0.0.0", port=8080)

else:
    print("Starting server")




