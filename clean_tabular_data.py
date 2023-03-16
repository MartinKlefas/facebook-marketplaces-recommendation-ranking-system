import pandas as pd 
from geopy.geocoders import Nominatim

def geoCode( df: pd.DataFrame, addressColumns):
        geolocator = Nominatim(timeout=10, user_agent = "myGeolocator")
        
        if isinstance(addressColumns, str):
                df['full_address'] = df[addressColumns]
        else:
                fullAddress = ""
                for column in addressColumns:
                        df['full_address'] += df[column]
 

        
        
        df['gcode'] = df.full_address.apply(geolocator.geocode)
        df['lat'] = [g.latitude for g in df.gcode]
        df['long'] = [g.longitude for g in df.gcode]

        return df

def clean_products(df: pd.DataFrame):        
        df["price"] = df["price"].str.replace('£', '', regex=False)
        df["price"] = df["price"].str.replace(',', '', regex=False)
        df['price'] = pd.to_numeric(df['price'],errors="coerce")
        df = geoCode(df=df,addressColumns="location")
        df = df.drop(columns=["Unnamed: 0",'full_address',"location"])

        return df

def clean_image_table(df: pd.DataFrame):       
        df = df.drop(columns=["Unnamed: 0"])

        return df

def import_image_data():
        df = pd.read_csv(filepath_or_buffer='Images.csv',lineterminator="\n")
        df = clean_image_table(df)

        return df

def import_product_data():
        df = pd.read_csv(filepath_or_buffer='Products.csv',lineterminator="\n")
        df = clean_products(df)

        return df
