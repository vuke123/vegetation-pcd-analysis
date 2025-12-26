#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

static void ensure_dir(const std::string& path)
{
    ::mkdir(path.c_str(), 0755); // ignores "already exists"
}

static std::string nowIso8601()
{
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return std::string(buf);
}

static std::string cmTag(float meters)
{
    int cm = static_cast<int>(std::lround(meters * 100.0f));
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << cm << "cm";
    return oss.str();
}

static bool loadPCD(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PCDReader reader;
    if (reader.read(filename, *cloud) == -1)
    {
        std::cerr << "Error reading PCD: " << filename << "\n";
        return false;
    }
    std::cout << "Loaded PCD: " << filename << " (" << cloud->size() << " pts)\n";
    return true;
}

int main()
{
    const std::string out_dir = "out_cluster";
    ensure_dir(out_dir);

    // ----------------------------------------------------------------------
    // INPUT: this should be NON-GROUND point cloud produced by your RANSAC app
    // Example: out_ground/nonground_leaf20cm_dist08cm_FINAL.pcd
    // ----------------------------------------------------------------------
    const std::string input_nonground_pcd =
        "./out_ground/nonground_leaf22cm_dist10cm_FINAL.pcd";  // <-- change to your file

    // Save downsampled variants (optional) + clusters
    const bool SAVE_DOWNSAMPLED = true;
    const bool SAVE_CLUSTERS    = true;

    // Parameter sweep
    std::vector<float> leaf_sizes         = {0.00f}; // 0.00 means "no voxel at all"
    std::vector<float> cluster_tolerances = {0.05f, 0.1f, 0.3f, 0.6f, 0.8f, 1.0f, 1.5f, 2.0f};

    // Clustering constraints (adjust for your vineyard density)
    const int MIN_CLUSTER_SIZE = 100;
    const int MAX_CLUSTER_SIZE = 450000;

    // Log
    std::ofstream log(out_dir + "/clustering_only_log.csv", std::ios::out);
    log << "timestamp,config_id,leaf_m,tol_m,"
           "input_pts,voxel_pts,clusters,valid,reason,"
           "voxel_s,cluster_s,total_s,"
           "input_file,downsample_file\n";

    pcl::PointCloud<pcl::PointXYZ>::Ptr input(new pcl::PointCloud<pcl::PointXYZ>);
    if (!loadPCD(input_nonground_pcd, input))
        return -1;

    pcl::PCDWriter writer;

    int config_id = 0;
    const auto global_start = std::chrono::high_resolution_clock::now();

    for (float leaf : leaf_sizes)
    {
        // 1) Optional voxel downsample
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_work(new pcl::PointCloud<pcl::PointXYZ>);
        double voxel_s = 0.0;

        if (leaf > 0.0f)
        {
            auto t0 = std::chrono::high_resolution_clock::now();

            pcl::VoxelGrid<pcl::PointXYZ> vg;
            vg.setInputCloud(input);
            vg.setLeafSize(leaf, leaf, leaf);
            vg.filter(*cloud_work);

            auto t1 = std::chrono::high_resolution_clock::now();
            voxel_s = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0;

            std::cout << "\n[VOXEL] leaf=" << leaf
                      << " | " << input->size() << " -> " << cloud_work->size()
                      << " pts | " << voxel_s << " s\n";
        }
        else
        {
            *cloud_work = *input; // no voxel
            std::cout << "\n[VOXEL] leaf=0 (disabled) | pts=" << cloud_work->size() << "\n";
        }

        if (cloud_work->size() < 100)
        {
            std::cout << "Too few points, skip leaf.\n";
            continue;
        }

        std::string downsample_file = "";
        if (SAVE_DOWNSAMPLED && leaf > 0.0f)
        {
            downsample_file = out_dir + "/cluster_input_leaf" + cmTag(leaf) + ".pcd";
            std::cout << "[Save] " << downsample_file << "\n";
            writer.write<pcl::PointXYZ>(downsample_file, *cloud_work, true);
        }

        // Build kd-tree ONCE per leaf
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_work);

        for (float tol : cluster_tolerances)
        {
            config_id++;

            // Guard: too-large tolerance vs spacing can explode runtime
            if (leaf > 0.0f && tol > 8.0f * leaf)
            {
                log << nowIso8601() << "," << config_id << ","
                    << leaf << "," << tol << ","
                    << input->size() << "," << cloud_work->size() << ","
                    << 0 << "," << 0 << ","
                    << "skip_tol_vs_leaf,"
                    << voxel_s << ",0,0,"
                    << input_nonground_pcd << "," << downsample_file << "\n";
                log.flush();

                std::cout << "[Skip] config" << config_id
                          << " tol=" << tol << " too large vs leaf=" << leaf << "\n";
                continue;
            }

            std::cout << "\n---------------------------------------------\n";
            std::cout << "CONFIG " << config_id
                      << " | leaf=" << leaf
                      << " | tol=" << tol << "\n";
            std::cout << "---------------------------------------------\n";

            auto c0 = std::chrono::high_resolution_clock::now();

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(tol);
            ec.setMinClusterSize(MIN_CLUSTER_SIZE);
            ec.setMaxClusterSize(MAX_CLUSTER_SIZE);
            ec.setSearchMethod(tree);
            ec.setInputCloud(cloud_work);
            ec.extract(cluster_indices);

            auto c1 = std::chrono::high_resolution_clock::now();
            double cluster_s =
                std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count() / 1000.0;

            const bool valid = (cluster_indices.size() <= 30);
            const std::string reason = valid ? "ok" : "too_many_clusters";

            std::cout << "clusters=" << cluster_indices.size()
                      << " | time=" << cluster_s << " s"
                      << (valid ? " | VALID\n" : " | INVALID\n");

            // Save clusters
            if (SAVE_CLUSTERS && valid)
            {
                int j = 0;
                for (const auto& cl : cluster_indices)
                {
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
                    cluster->points.reserve(cl.indices.size());

                    for (int idx : cl.indices)
                        cluster->points.push_back((*cloud_work)[idx]);

                    cluster->width = static_cast<uint32_t>(cluster->size());
                    cluster->height = 1;
                    cluster->is_dense = true;

                    std::ostringstream ss;
                    ss << out_dir << "/config" << config_id
                       << "_leaf" << cmTag(leaf)
                       << "_tol" << cmTag(tol)
                       << "_cluster_" << std::setw(2) << std::setfill('0') << j << ".pcd";
                    writer.write<pcl::PointXYZ>(ss.str(), *cluster, true);
                    ++j;
                }
            }

            auto now = std::chrono::high_resolution_clock::now();
            double total_s =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - global_start).count() / 1000.0;

            log << nowIso8601() << "," << config_id << ","
                << leaf << "," << tol << ","
                << input->size() << "," << cloud_work->size() << ","
                << cluster_indices.size() << ","
                << (valid ? 1 : 0) << ","
                << reason << ","
                << voxel_s << "," << cluster_s << "," << total_s << ","
                << input_nonground_pcd << "," << downsample_file << "\n";
            log.flush();
        }
    }

    std::cout << "\nDone. Log: " << out_dir + "/clustering_only_log.csv" << "\n";
    return 0;
}
