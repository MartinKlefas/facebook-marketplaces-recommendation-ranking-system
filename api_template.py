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
import faiss, os, shutil, uuid
import numpy as np

from contextlib import asynccontextmanager

import image_processor


class FeatureExtractor(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(FeatureExtractor, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = torch.load('model_evaluation/feature/best_weights.pt')
        
        self.decoder = decoder

    def forward(self, image):
        x = self.main(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str



try:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load('model_evaluation/feature/best_weights.pt')
    pass
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:
    index = faiss.read_index('images_faiss.index')
    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

@asynccontextmanager
async def lifespan(app: FastAPI):
    #everything else on startup is already implemented
    image_processor.validate_output_dir("received_files")
    
    yield
    # Clean up the uploads folder
    for filename in os.listdir('received_files'):
        file_path = os.path.join('received_files', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    print("uploads directory cleared. Bye!")

app = FastAPI(lifespan=lifespan)
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

@app.post('/healthcheck/file')
def healthcheck_file(image: UploadFile = File(...)):
  newPath = os.path.join('received_files',str(uuid.uuid4())+'_'+image.filename)
  with open(newPath, "wb") as buffer:
      shutil.copyfileobj(image.file, buffer)

  return {"message": newPath}

  
@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    newPath = os.path.join('received_files',str(uuid.uuid4())+'_'+image.filename)
    with open(newPath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    embedding = image_processor.getEmbedding(image=newPath,model=model)

    return JSONResponse(content={
    "features": embedding.tolist(), 
   
        })
  
@app.post('/predict/similar_images')
def predict_combined(matches: str = 3,image: UploadFile = File(...)):

    newPath = os.path.join('received_files',str(uuid.uuid4())+'_'+image.filename)
    with open(newPath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    searchEmbedding = image_processor.getEmbedding(newPath,model=model)
    searchEmbedding = np.array(searchEmbedding,dtype="float32",ndmin=2)

    nprobe = 5
    distances, indices = index.search(x=searchEmbedding,k=matches)

    return JSONResponse(content={
    "similar_indices": indices.tolist(), # Return the index of similar images here
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)


