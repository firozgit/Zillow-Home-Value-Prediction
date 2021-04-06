import numpy as np
import pandas as pd


"""get_data definition to do the following

1. Read the train/properties data
2. Merge the datasets
3. Remove the duplicate parcels
4. Convert the categorical features to "object" data type.


 """

def read_data(data_string):
    """
    Read the train/properties data set
    
    Parameters:
    data_string -- "train_2016",  "train_2017", "properties_2016", "properties_2017"
    
    
    Returns:
    df - dataframe from the "data/raw" folder
    
    
    """
    df = pd.read_csv("../data/raw/{0}.csv".format(data_string))
    
    return df



def remove_duplicate_parcels(df):    
    """
    Remove the records with duplicate parcelid in the merged train data set.
    
    Parameters:
    df -- merged data frame
    
    Returns:    
    unique_df -- a dataframe with unique parcelid's
    
    """    
    parcel_count = df.groupby(["parcelid"]).size()
    unique_parcel = df[df["parcelid"].isin(parcel_count[parcel_count == 1].index)]
    duplicated_parcel = df[df["parcelid"].isin(parcel_count[parcel_count > 1].index)]
    duplicated_parcel_made_unique = duplicated_parcel.sample(frac=1, random_state=42).groupby(["parcelid"]).head(1)
    unique_df = pd.concat([unique_parcel, duplicated_parcel_made_unique], axis=0)
    
    return unique_df
    

def get_data(data_string):
    """
    Read the train/test dataset and merge with properties data set and remove duplicate parcelid's in train
    
    Parameters:
    data_string -- "train" or "test" 
    
    Returns:
    X, y -- a tuple of dataframe X and Series y
    
    
    """         
    year = 2016 if data_string == "train" else 2017
        
    train = read_data("train_{0}".format(year))
    properties = read_data("properties_{0}".format(year))
    merged = pd.merge(train, properties, on="parcelid", how="left")
                      
    if data_string == "train":
        merged = remove_duplicate_parcels(merged)
                          
    y = merged["logerror"]                          
    merged = merged.drop(columns=["logerror"], axis=1) 
    
    id_col = ["parcelid"]
    cat_cols = ["airconditioningtypeid", "architecturalstyletypeid", "buildingclasstypeid", 
                "buildingqualitytypeid", "decktypeid", "fips", "fireplaceflag", 
                "hashottuborspa", "heatingorsystemtypeid", "pooltypeid10", "pooltypeid2",
                "pooltypeid7", "propertycountylandusecode", "propertylandusetypeid", 
                "propertyzoningdesc", "rawcensustractandblock", "censustractandblock", 
                "regionidcounty", "regionidcity", "regionidzip", "regionidneighborhood", 
                "storytypeid", "typeconstructiontypeid", "yearbuilt", "assessmentyear", 
                "taxdelinquencyflag", "taxdelinquencyyear"]

    # convert all categorical variables to categorical type
    for col in (id_col + cat_cols):
        merged[col] = merged[col].astype("object")

    # convert transactiondate column to datetime type    
    merged["transactiondate"] = pd.to_datetime(merged["transactiondate"])    

    return merged, y