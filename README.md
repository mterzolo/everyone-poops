# Everyone Poops
This project takes in 311 call request data to clean up human waste and predicts where and when people might poop on the streets of San Francisco.

## Getting Started
You will need to download some data in order to get this project to run:

311 Data

Link
https://data.sfgov.org/api/views/vw6y-z8j6/rows.csv?accessType=DOWNLOAD

Put into
/data/sf_open/

Land Use Data

Link
https://data.sfgov.org/api/geospatial/us3s-fp9q?method=export&format=Shapefile

Put into
/data/sf_open

Census Block Data

Link
https://data.sfgov.org/api/geospatial/rarb-5ahf?method=export&format=Shapefile

Put into
/data/sf_open

Demographic Data
Demographic data can be found at https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml
Data should be extracted at the most granular geographic level (census block groups).

Once the data is downloaded and put into its respective locations, run each script in order (1_clean_from_source.ipynb, then 2-0_make_blocks.ipynb, etc.)
