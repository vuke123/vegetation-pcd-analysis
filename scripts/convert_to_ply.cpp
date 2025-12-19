#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input.pcd> <output.ply>" << std::endl;
        return -1;
    }
    
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    
    // Load PCD file
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(inputFile, *cloud) == -1) {
        std::cerr << "Error loading PCD file: " << inputFile << std::endl;
        return -1;
    }
    
    std::cout << "Loaded " << cloud->points.size() << " points from " << inputFile << std::endl;
    
    // Save as PLY
    if (pcl::io::savePLYFileBinary(outputFile, *cloud) == -1) {
        std::cerr << "Error saving PLY file: " << outputFile << std::endl;
        return -1;
    }
    
    std::cout << "Saved to " << outputFile << std::endl;
    
    return 0;
}
