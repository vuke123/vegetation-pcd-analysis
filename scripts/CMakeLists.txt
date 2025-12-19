cmake_minimum_required(VERSION 3.10)
project(PointCloudClustering)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PCL package (removed visualization component - using Python instead)
find_package(PCL 1.8 REQUIRED COMPONENTS common io filters kdtree segmentation)

# Find PDAL package
find_package(PDAL REQUIRED)

# Include directories
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${PDAL_INCLUDE_DIRS})

# Link directories
link_directories(${PCL_LIBRARY_DIRS})

# Add definitions
add_definitions(${PCL_DEFINITIONS})

# Create converter executable
add_executable(convert_to_ply convert_to_ply.cpp)

# Link libraries for converter
target_link_libraries(convert_to_ply 
    ${PCL_LIBRARIES}
)

add_executable(pcl_tutorial_clustering pcl_tutorial_clustering.cpp)
target_link_libraries(pcl_tutorial_clustering ${PCL_LIBRARIES} ${PDAL_LIBRARIES})