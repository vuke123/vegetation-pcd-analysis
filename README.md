# Vegetation-Metrics-Estimation
Estimation of NDVI indexes and volume in vegetation (vineyards and olive trees)

Step 1) : 

Idea : Load data and try to segment the plants using instance segmentation with DBSCAN to extract the parts of the point cloud that correspond to plants.

Current state : Found the way to transform point cloud to image format but needs to convert cloud to 3chanel rgb because currently is just rasterized over z axis and doesn't contain information about color.
(pcd_to_jpg.ipynb file).

Step 2): 

Idea : Ground removal 

Current state : Can be done with RANSAC ground segmentation (currently implemented in code but the result isn't optimal and doesn't satisfy needs). Easier approach is to use height over ground feature from CloudCompare tool.

Step 3) : 

 
Idea : Display differences between LIDAR, IR and MS point cloud.
Voxelize object for robust volume calculation. 

Current state: 

Voxelized manually extracted (CloudCompare + height over ground filtering) plants. 
Done with custom code and also with Open3D 0.19.0. library.
(read_plot_voxelization.ipynb) 

Step 4) : 

Idea : Volume calculation. Through voxelization for robust and aproximative method and slicing method for more detailed version.

Current state : 

Both logic are implemented. Just need data with local coordinates so I can compute volume in correct metrics. Afterwards, I should know if my implementations work fine. 
(volume_calculation.ipynb)

Step 5) : 

Idea : NDVI indexes calculation 

Current state : In development. Saving GeoTIFF file after computation and visualizing it in QGIS Desktop (rasterizing.ipynb).
