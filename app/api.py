import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
import faiss, os, shutil, uuid, time
import numpy as np
from datetime import datetime
import pandas as pd

from contextlib import asynccontextmanager

import image_processor



start_time = datetime.now()

try:
    
    device = torch.device("cpu")


    model = torch.load('best_weights.pt',map_location=torch.device('cpu'))
    
    pass
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:
    index = faiss.read_index('images_faiss.index')
    pass
except:
    raise OSError("No FAISS index found. Check that you have the encoder and the model in the correct location")

try:
    imageList =   pd.read_csv(filepath_or_buffer='training_data.csv')
    pass
except:
    raise OSError("No image key dictionary found. Check that it's in the correct folder")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  timediff = datetime.now() - start_time


  msg = "API has been running for " + str(timediff )
  
  return {"message": msg}

@app.post('/healthcheck/file')
def healthcheck_file(image: UploadFile = File(...)):
  newPath = os.path.join(str(uuid.uuid4())+'_'+image.filename)
  try:
      with open(newPath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
      os.unlink(newPath)  
      return {"message": newPath}
  except Exception as ex:
      return {"message" : ex}
  

  
@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    newPath = os.path.join(str(uuid.uuid4())+'_'+image.filename)#'received_files',str(uuid.uuid4())+'_'+image.filename)
    with open(newPath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    embedding = image_processor.getEmbedding(image=newPath,model=model)

    return JSONResponse(content={
    "features": embedding.tolist(), 
   
        })
  
@app.post('/predict/similar_image_indices')
def predict_combined(matches: str = 3,image: UploadFile = File(...)):

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

@app.post('/predict/similar_images')
def return_images(matches: str = 3,image: UploadFile = File(...)):

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


