#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/angles.h>

#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/Options.hpp>
#include <pdal/io/LasReader.hpp>
#include <pdal/io/LasHeader.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

// Median in-place (nth_element). v is modified.
static double medianOf(std::vector<double>& v)
{
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();

    const size_t n = v.size();
    const size_t mid = n / 2;

    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double med = v[mid];

    if (n % 2 == 0)
    {
        auto maxLower = *std::max_element(v.begin(), v.begin() + mid);
        med = 0.5 * (maxLower + med);
    }
    return med;
}

// Local timestamp for logs.
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

// 0.05 -> "05cm" for filenames.
static std::string cmTag(float meters)
{
    int cm = static_cast<int>(std::lround(meters * 100.0f));
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << cm << "cm";
    return oss.str();
}

// Read .pcd via PCL, .las via PDAL.
// Note: PDAL returns scaled+offset coordinates.
bool loadPointCloud(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    const std::string extension = filename.substr(filename.find_last_of(".") + 1);

    if (extension == "pcd" || extension == "PCD")
    {
        pcl::PCDReader reader;
        if (reader.read(filename, *cloud) == -1)
        {
            std::cerr << "Error reading PCD: " << filename << "\n";
            return false;
        }
        std::cout << "Loaded PCD: " << filename << "\n";
        return true;
    }

    if (extension == "las" || extension == "LAS")
    {
        try
        {
            pdal::Options options;
            options.add("filename", filename);
            options.add("nosrs", true);

            pdal::PointTable table;
            pdal::LasReader las_reader;
            las_reader.setOptions(options);

            las_reader.prepare(table);
            pdal::PointViewSet viewSet = las_reader.execute(table);
            pdal::PointViewPtr view = *viewSet.begin();

            // Header scale/offset (useful for debugging units).
            pdal::LasHeader h = las_reader.header();
            const double sx = h.scaleX(), sy = h.scaleY(), sz = h.scaleZ();
            const double ox = h.offsetX(), oy = h.offsetY(), oz = h.offsetZ();

            std::cout << "Loading LAS: " << filename << "\n";
            std::cout << "  Points: " << view->size() << "\n";
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  Scales : (" << sx << ", " << sy << ", " << sz << ")\n";
            std::cout << "  Offsets: (" << ox << ", " << oy << ", " << oz << ")\n";

            // Quick median sanity check (sampled).
            const size_t n = static_cast<size_t>(view->size());
            const size_t maxSamples = 1'000'000;
            const size_t step = std::max<size_t>(1, n / maxSamples);

            std::vector<double> xs, ys, zs;
            xs.reserve(std::min(n, maxSamples));
            ys.reserve(std::min(n, maxSamples));
            zs.reserve(std::min(n, maxSamples));

            for (pdal::PointId i = 0; i < view->size(); i += step)
            {
                xs.push_back(view->getFieldAs<double>(pdal::Dimension::Id::X, i));
                ys.push_back(view->getFieldAs<double>(pdal::Dimension::Id::Y, i));
                zs.push_back(view->getFieldAs<double>(pdal::Dimension::Id::Z, i));
            }

            std::cout << "  Median X,Y,Z: ("
                      << medianOf(xs) << ", " << medianOf(ys) << ", " << medianOf(zs) << ")\n";

            // Copy into PCL cloud.
            cloud->clear();
            cloud->points.reserve(view->size());

            for (pdal::PointId i = 0; i < view->size(); ++i)
            {
                pcl::PointXYZ p;
                p.x = static_cast<float>(view->getFieldAs<double>(pdal::Dimension::Id::X, i));
                p.y = static_cast<float>(view->getFieldAs<double>(pdal::Dimension::Id::Y, i));
                p.z = static_cast<float>(view->getFieldAs<double>(pdal::Dimension::Id::Z, i));
                cloud->points.push_back(p);
            }

            cloud->width = static_cast<uint32_t>(cloud->size());
            cloud->height = 1;
            cloud->is_dense = true;

            std::cout << "Loaded LAS: " << cloud->size() << " points\n";
            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error reading LAS: " << e.what() << "\n";
            return false;
        }
    }

    std::cerr << "Unsupported format: " << extension << " (use .pcd or .las)\n";
    return false;
}

static void ensure_dir(const std::string& path)
{
    // Creates directory if missing. Ignores "already exists".
    ::mkdir(path.c_str(), 0755);
}

int main()
{
    // Output folder + log.
    const std::string out_dir = "out";
    ensure_dir(out_dir);

    const bool SAVE_ORIGINAL      = true;
    const bool SAVE_DOWNSAMPLED   = true;
    const bool SAVE_NONGROUND     = true;
    const bool SAVE_GROUND        = false;

    std::ofstream log(out_dir + "/clustering_log.csv", std::ios::out);
    log << "timestamp,config_id,leaf_m,dist_m,tol_m,"
           "orig_pts,voxel_pts,ground_pts,nonground_pts,"
           "clusters,valid,reason,"
           "voxel_s,plane_s,cluster_s,total_s,"
           "downsample_file,nonground_file\n";

    // Load input cloud.
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    const std::string input_file = "../datasource/LOCAL_VINEYARD_MS.las";

    if (!loadPointCloud(input_file, cloud))
    {
        std::cerr << "Failed to load: " << input_file << "\n";
        return -1;
    }
    std::cout << "PointCloud loaded: " << cloud->size() << " points\n";

    pcl::PCDWriter writer;

    // Save original (binary).
    if (SAVE_ORIGINAL)
    {
        const std::string orig_path = out_dir + "/original_cloud.pcd";
        std::cout << "[Save] " << orig_path << "\n";
        writer.write<pcl::PointXYZ>(orig_path, *cloud, true);
    }

    // Parameter sweep (meters).
    std::vector<float> leaf_sizes          = {0.19f, 0.20f, 0.22f};
    std::vector<float> distance_thresholds = {0.02f, 0.8f, 0.12f};
    std::vector<float> cluster_tolerances  = {0.6f, 0.80f, 1.0f};

    const std::size_t L = leaf_sizes.size();
    const std::size_t D = distance_thresholds.size();
    const std::size_t T = cluster_tolerances.size();

    const auto global_start = std::chrono::high_resolution_clock::now();

    // config_id is deterministic:
    // config_id = li*(D*T) + di*(T) + ti + 1
    for (std::size_t li = 0; li < L; ++li)
    {
        const float leaf_size = leaf_sizes[li];

        // 1) Voxel downsample.
        const auto leaf_start = std::chrono::high_resolution_clock::now();
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_leaf(new pcl::PointCloud<pcl::PointXYZ>);
        vg.setInputCloud(cloud);
        vg.setLeafSize(leaf_size, leaf_size, leaf_size);
        vg.filter(*cloud_leaf);

        const auto leaf_end = std::chrono::high_resolution_clock::now();
        const double voxel_s =
            std::chrono::duration_cast<std::chrono::milliseconds>(leaf_end - leaf_start).count() / 1000.0;

        std::cout << "\n################################################################################\n";
        std::cout << "LEAF " << leaf_size << " -> " << cloud_leaf->size()
                  << " pts | voxel " << voxel_s << " s\n";
        std::cout << "################################################################################\n";

        if (cloud_leaf->size() < 100)
        {
            std::cout << "Too few points after voxel. Skip leaf.\n";
            continue;
        }

        // Save downsampled.
        const std::string downsample_file = out_dir + "/downsampled_leaf" + cmTag(leaf_size) + ".pcd";
        if (SAVE_DOWNSAMPLED)
        {
            std::cout << "[Save] " << downsample_file << "\n";
            writer.write<pcl::PointXYZ>(downsample_file, *cloud_leaf, true);
        }

        // 2) Ground plane per distance threshold.
        for (std::size_t di = 0; di < D; ++di)
        {
            const float dist_thresh = distance_thresholds[di];
            const auto plane_start = std::chrono::high_resolution_clock::now();

            pcl::SACSegmentation<pcl::PointXYZ> seg;
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);

            seg.setOptimizeCoefficients(false);
            seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setMaxIterations(50);
            seg.setDistanceThreshold(dist_thresh);
            seg.setAxis(Eigen::Vector3f(0.f, 0.f, 1.f));  // Z-up
            seg.setEpsAngle(pcl::deg2rad(10.0f));

            seg.setInputCloud(cloud_leaf);
            seg.segment(*inliers, *coeffs);

            // If plane fails, log all tolerances as skipped.
            if (inliers->indices.empty() || coeffs->values.size() < 4)
            {
                for (std::size_t ti = 0; ti < T; ++ti)
                {
                    const int config_id = static_cast<int>(li * (D * T) + di * T + ti + 1);
                    log << nowIso8601() << "," << config_id << ","
                        << leaf_size << "," << dist_thresh << "," << cluster_tolerances[ti] << ","
                        << cloud->size() << "," << cloud_leaf->size() << ","
                        << 0 << "," << 0 << ","
                        << 0 << "," << 0 << ","
                        << "no_plane,"
                        << voxel_s << ",0,0,0,"
                        << downsample_file << ",\n";
                }
                std::cout << "No plane for dist=" << dist_thresh << ". Skip.\n";
                continue;
            }

            // Plane coefficients: ax + by + cz + d = 0.
            const float a = coeffs->values[0];
            const float b = coeffs->values[1];
            const float c = coeffs->values[2];
            const float d = coeffs->values[3];
            const float denom = std::sqrt(a*a + b*b + c*c);

            // Split ground vs non-ground by point-to-plane distance.
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nonground(new pcl::PointCloud<pcl::PointXYZ>);
            cloud_ground->points.reserve(cloud_leaf->size() / 2);
            cloud_nonground->points.reserve(cloud_leaf->size() / 2);

            for (const auto& p : cloud_leaf->points)
            {
                const float dist = std::fabs(a*p.x + b*p.y + c*p.z + d) / denom;
                (dist <= dist_thresh) ? cloud_ground->points.push_back(p)
                                      : cloud_nonground->points.push_back(p);
            }

            cloud_ground->width = static_cast<uint32_t>(cloud_ground->size());
            cloud_ground->height = 1;
            cloud_ground->is_dense = true;

            cloud_nonground->width = static_cast<uint32_t>(cloud_nonground->size());
            cloud_nonground->height = 1;
            cloud_nonground->is_dense = true;

            const auto plane_end = std::chrono::high_resolution_clock::now();
            const double plane_s =
                std::chrono::duration_cast<std::chrono::milliseconds>(plane_end - plane_start).count() / 1000.0;

            std::cout << "\n[Ground] leaf=" << leaf_size
                      << " dist=" << dist_thresh
                      << " | ground=" << cloud_ground->size()
                      << " nonground=" << cloud_nonground->size()
                      << " | " << plane_s << " s\n";

            // Save non-ground per (leaf,dist).
            const std::string nonground_file =
                out_dir + "/nonground_leaf" + cmTag(leaf_size) + "_dist" + cmTag(dist_thresh) + ".pcd";

            if (SAVE_NONGROUND)
            {
                std::cout << "[Save] " << nonground_file << "\n";
                writer.write<pcl::PointXYZ>(nonground_file, *cloud_nonground, true);
            }

            if (SAVE_GROUND)
            {
                const std::string ground_file =
                    out_dir + "/ground_leaf" + cmTag(leaf_size) + "_dist" + cmTag(dist_thresh) + ".pcd";
                std::cout << "[Save] " << ground_file << "\n";
                writer.write<pcl::PointXYZ>(ground_file, *cloud_ground, true);
            }

            // If nothing left, log all tolerances as skipped.
            if (cloud_nonground->empty())
            {
                for (std::size_t ti = 0; ti < T; ++ti)
                {
                    const int config_id = static_cast<int>(li * (D * T) + di * T + ti + 1);
                    log << nowIso8601() << "," << config_id << ","
                        << leaf_size << "," << dist_thresh << "," << cluster_tolerances[ti] << ","
                        << cloud->size() << "," << cloud_leaf->size() << ","
                        << cloud_ground->size() << "," << cloud_nonground->size() << ","
                        << 0 << "," << 0 << ","
                        << "empty_nonground,"
                        << voxel_s << "," << plane_s << ",0,0,"
                        << downsample_file << "," << nonground_file << "\n";
                }
                std::cout << "Empty non-ground. Skip clustering.\n";
                continue;
            }

            // Reuse kd-tree for all tolerances.
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
            tree->setInputCloud(cloud_nonground);

            // 3) Euclidean clustering per tolerance.
            for (std::size_t ti = 0; ti < T; ++ti)
            {
                const float clust_tol = cluster_tolerances[ti];
                const int config_id = static_cast<int>(li * (D * T) + di * T + ti + 1);

                // Guard: huge tol on dense clouds can be very slow.
                if (clust_tol > 8.0f * leaf_size)
                {
                    log << nowIso8601() << "," << config_id << ","
                        << leaf_size << "," << dist_thresh << "," << clust_tol << ","
                        << cloud->size() << "," << cloud_leaf->size() << ","
                        << cloud_ground->size() << "," << cloud_nonground->size() << ","
                        << 0 << "," << 0 << ","
                        << "skip_tol_vs_leaf,"
                        << voxel_s << "," << plane_s << ",0,0,"
                        << downsample_file << "," << nonground_file << "\n";

                    std::cout << "[Skip] config" << config_id
                              << " tol too large vs leaf\n";
                    continue;
                }

                std::cout << "\n--------------------------------------------------------------------------------\n";
                std::cout << "CONFIG " << config_id
                          << " | leaf=" << leaf_size
                          << " dist=" << dist_thresh
                          << " tol=" << clust_tol << "\n";
                std::cout << "--------------------------------------------------------------------------------\n";

                const auto cluster_start = std::chrono::high_resolution_clock::now();

                std::vector<pcl::PointIndices> cluster_indices;
                pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
                ec.setClusterTolerance(clust_tol);
                ec.setMinClusterSize(1000);
                ec.setMaxClusterSize(450000);
                ec.setSearchMethod(tree);
                ec.setInputCloud(cloud_nonground);
                ec.extract(cluster_indices);

                const auto cluster_end = std::chrono::high_resolution_clock::now();
                const double cluster_s =
                    std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start).count() / 1000.0;

                const bool valid = (cluster_indices.size() <= 30);
                const std::string reason = valid ? "ok" : "too_many_clusters";

                std::cout << "  clusters=" << cluster_indices.size()
                          << " | " << cluster_s << " s"
                          << (valid ? " | VALID" : " | INVALID") << "\n";

                // Save clusters only for valid configs.
                if (valid)
                {
                    int j = 0;
                    for (const auto& cluster : cluster_indices)
                    {
                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
                        cloud_cluster->points.reserve(cluster.indices.size());

                        for (int idx : cluster.indices)
                            cloud_cluster->points.push_back((*cloud_nonground)[idx]);

                        cloud_cluster->width = static_cast<uint32_t>(cloud_cluster->size());
                        cloud_cluster->height = 1;
                        cloud_cluster->is_dense = true;

                        std::ostringstream ss;
                        ss << out_dir << "/config" << config_id << "_cluster_"
                           << std::setw(2) << std::setfill('0') << j << ".pcd";
                        writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster, true);
                        ++j;
                    }
                }

                const auto now = std::chrono::high_resolution_clock::now();
                const double total_s =
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - global_start).count() / 1000.0;

                // Log one row per (leaf,dist,tol).
                log << nowIso8601() << "," << config_id << ","
                    << leaf_size << "," << dist_thresh << "," << clust_tol << ","
                    << cloud->size() << "," << cloud_leaf->size() << ","
                    << cloud_ground->size() << "," << cloud_nonground->size() << ","
                    << cluster_indices.size() << ","
                    << (valid ? 1 : 0) << ","
                    << reason << ","
                    << voxel_s << "," << plane_s << "," << cluster_s << "," << total_s << ","
                    << downsample_file << "," << nonground_file << "\n";

                log.flush();
            }
        }
    }

    std::cout << "\nDone. Log: " << out_dir + "/clustering_log.csv" << "\n";
    return 0;
}
