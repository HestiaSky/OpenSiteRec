# OpenSiteRec

### Usage

Each folder of Chicago, NYC, Singapore and Tokyo contains the OpenSiteRec data of the city.
In each folder, the structure is as follows:

- City/
	- City_KG_plus.pkl 
    	- the main file that each instance denote a POI with corresponding information
	- City_Region.geojson 
    	- the geojson file that collect all the regions as Polygons in the city
	- City_BA.geojson 
    	- the geojson file that collect all the business areas as Polygons in the city
	- City_brands.pkl 
    	- the relation file that represent the relations between brands
	- City_regions.pkl 
    	- the relation file that represent the relations between regions

Please use the python package Pandas and GeoPandas to load the file:

    import pandas as pd
    import geopandas as gpd
    
    df = pd.read_pickle('xxx.pkl')
    gdf = gpd.read_file('xxx.geojson')
    
The folder baseline contains the implementation code of the baselines.

We also provide a demo data of Singapore using .csv format in folder demo.
