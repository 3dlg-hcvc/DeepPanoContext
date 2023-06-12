The DeepPanoContext project uses iGibson 1.0 and its dateset. At the time, the dataset is unencrypted, and can be used to generated the proposed new dataset with rendered images, formatted layout and object GTs.

However, after an update of iGibson, the dataset was replaced with an encrypted one which is not compatible with the old version 1.0.3.

If you would like to preprocess the data by yourself, please decompress the unencrypted iGibson data ```data.zip``` into ``` HOME_DIRECTORY\anaconda3\envs\Pano3D\lib\python3.7\site-packages\gibson2\``` like ``` HOME_DIRECTORY\anaconda3\envs\Pano3D\lib\python3.7\site-packages\gibson2\data```. Then follow the README of DeepPanoContext to prepare the dataset.

You can also skip data preprocessing by decompressing the preprocessed dataset ```igibson``` and ```igibson_obj``` into ```PROJECT_FOLDER\data``` like:
```
PROJECT_FOLDER\data\igibson
PROJECT_FOLDER\data\igibson_obj
```
