
# Facebook Marketplace Recommendation Ranking System

This is one of the specialisation projects from the AICore curriculum. Some details of the (vastly simplified version of the) system being cloned are explained in this video here: https://www.youtube.com/watch?v=1Z5V2VrHTTA

The critical part of this is determining the products "similar" to those in which a facebook user has already shown an interest in. We will tackle part of this problem in this project by finding product photos "similar" to any searched photo. The approach to tackling this problem is as follows:  
# Part 1

1) Connect to EC2 to download sample data, import and clean it.
	- tabular data cleaning done on [a Jupyter Notebook](https://github.com/MartinKlefas/facebook-marketplaces-recommendation-ranking-system/blob/fa2934bf4db916fd06c2b923ba89367a97ad098a/Data_Import_&_Clean.ipynb) to test options quickly
		- chosen methods then refactored into `clean_tabular_data.py`
	- image cleaning done using the Pillow library
		- Images had different bitrate and depths
		- Resizing and saving with fixed options to standardise this
		- code implemented in `clean_images.py`
2) Convert `Product Categories` into machine readable integers with a dictionary to act as a human readable key.

# Part 2
Create a classification model & feature extraction model - [implemented in a notebook as a one-off exploration](https://github.com/MartinKlefas/facebook-marketplaces-recommendation-ranking-system/blob/main/retrain.ipynb)
1) Import data into PyTorch
2) Strip the output layer from ResNet 50 and replace with a retrained layer
3) Create a pre-processor script for future images [image_processor.py](https://github.com/MartinKlefas/facebook-marketplaces-recommendation-ranking-system/blob/main/image_processor.py)
4) Verify that the CNN mostly correctly predicts what category an item will be in
   - this indicates that the CNN is properly partitioning the images and so is identifying similarities properly.

# Part 3
FAISS is a relational database useful for finding vectors similar to the search term. If you use an encoder to "vectorise" a collection of any kind of object, you can then use a FAISS database to find objects similar to those in the database. In this implementation we're encoding the product images, with the feature extraction model. Putting the encoded data into FAISS allows us to then test an image (novel or existing) to see which other images are "similar" to it, and so which other products to show to the user.

In order to do this we [implemented a notebook](https://github.com/MartinKlefas/facebook-marketplaces-recommendation-ranking-system/blob/main/faiss.ipynb) that:
1. Vectorised all samples
2. Added them to FAISS and select an appropriate function to "index" them


We then create a lookup function to return details of the most "similar" results when this database is searched, this is in the notebook too for testing, but implemented in fixed python in the next part

# Part 4
The most apt way to interact with this new database is via an API - as this cuts down execution time for future scripts, allows the database to be centrally updated if needed, and minimises the transmission of potentially sensitive (and definately very large) data.
It's less of an issue in such a small sample project, but in the real world this FAISS implementation and the PyTorch model behind it would also need to be hosted on a specialised and costly server with a potentially large number of specialised GPUs to make the database responsive. Making an API thereby cuts down on hardware costs for end users who also need this information.

1) Migrate key methods into a `fastapi` based API wrapper
2) Test the API locally using `uvicorn`
3) Create a docker file for the implementation, adjusting the configuration for the differences between localhost and the target EC2 Instance:
    - No GPU on this instance
    - Limited local storage
    - Linux vs Windows package source differences
        - pyTorch is GPU by default on Linux and CPU by default on windows for instance
4) migrate docker to EC2 and test.
 
> Written with [StackEdit](https://stackedit.io/).