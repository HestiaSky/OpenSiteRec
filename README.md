# OpenSiteRec

![image](https://github.com/HestiaSky/OpenSiteRec/blob/main/fig/Schema.png)

### Usage

Each folder of Chicago, NYC, Singapore and Tokyo contains the OpenSiteRec data of the city.
In each folder, the structure is as follows:

- City/
	- City_KG_plus.pkl/.csv
    	- the main file that each instance denote a POI with corresponding information
	- City_Region.geojson 
    	- the geojson file that collect all the regions as Polygons in the city
	- City_BA.geojson 
    	- the geojson file that collect all the business areas as Polygons in the city
	- City_brands.pkl/.csv 
    	- the relation file that represent the relations between brands
	- City_regions.pkl/.csv 
    	- the relation file that represent the relations between regions

Please use the python package Pandas and GeoPandas to load the file:

    import pandas as pd
    import geopandas as gpd
    
    df = pd.read_pickle('xxx.pkl')
    gdf = gpd.read_file('xxx.geojson')
    
The folder baseline contains the implementation code of the baselines.

We also provide a demo data of Singapore using .csv format in folder demo.

### Benchmarking Code
To facilitate the usage, we provide the code of baseline model training, where you can find the code to load the dataset in baseline/data_utils.py and to specify your training schedule in baseline/main.py.

However, this data loader merely organize the data in a simple and vanilla way.
It is encouraged to play with the data according to your own idea.