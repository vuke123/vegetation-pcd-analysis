#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/angles.h>
#include <pcl/filters/extract_indices.h>

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

// ---------- utilities ----------
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

static void ensure_dir(const std::string& path)
{
    ::mkdir(path.c_str(), 0755); // ignores "already exists"
}

// ---------- IO: PCD or LAS ----------
bool loadPointCloud(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    const std::string ext = filename.substr(filename.find_last_of(".") + 1);

    if (ext == "pcd" || ext == "PCD")
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

    if (ext == "las" || ext == "LAS")
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

            pdal::LasHeader h = las_reader.header();
            std::cout << "Loaded LAS: " << filename << "\n";
            std::cout << "  Points: " << view->size() << "\n";
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  Scales : (" << h.scaleX() << ", " << h.scaleY() << ", " << h.scaleZ() << ")\n";
            std::cout << "  Offsets: (" << h.offsetX() << ", " << h.offsetY() << ", " << h.offsetZ() << ")\n";

            // sampled median sanity check
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
            std::cout << "  Median X,Y,Z (sampled): (" << medianOf(xs) << ", " << medianOf(ys) << ", " << medianOf(zs) << ")\n";

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

            std::cout << "Loaded into PCL: " << cloud->size() << " pts\n";
            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error reading LAS: " << e.what() << "\n";
            return false;
        }
    }

    std::cerr << "Unsupported format: " << ext << " (use .pcd or .las)\n";
    return false;
}

int main()
{
    const std::string out_dir = "out_ground";
    ensure_dir(out_dir);

    // ---------- what to save ----------
    const bool SAVE_ORIGINAL    = false; 
    const bool SAVE_DOWNSAMPLED = true;
    const bool SAVE_NONGROUND   = true;
    const bool SAVE_GROUND      = false;

    // ---------- input ----------
    const std::string input_file = "../datasource/LOCAL_VINEYARD_MS.las";

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (!loadPointCloud(input_file, cloud))
    {
        std::cerr << "Failed to load: " << input_file << "\n";
        return -1;
    }

    pcl::PCDWriter writer;

    if (SAVE_ORIGINAL)
    {
        const std::string orig_path = out_dir + "/original_cloud.pcd";
        std::cout << "[Save] " << orig_path << "\n";
        writer.write<pcl::PointXYZ>(orig_path, *cloud, true);
    }

    // ---------- sweep params (meters) ----------
    std::vector<float> leaf_sizes          = {0.19f, 0.20f, 0.22f};
    std::vector<float> distance_thresholds = {0.02f, 0.08f, 0.12f};

    // Log per (leaf,dist)
    std::ofstream log(out_dir + "/ground_removal_log.csv", std::ios::out);
    log << "timestamp,case_id,leaf_m,dist_m,orig_pts,voxel_pts,ground_pts,nonground_pts,voxel_s,plane_s,"
           "downsample_file,nonground_file\n";

    int case_id = 0;

    for (float leaf : leaf_sizes)
    {
        // Voxel once per leaf
        auto t0 = std::chrono::high_resolution_clock::now();

        pcl::VoxelGrid<pcl::PointXYZ> vg;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_leaf(new pcl::PointCloud<pcl::PointXYZ>);
        vg.setInputCloud(cloud);
        vg.setLeafSize(leaf, leaf, leaf);
        vg.filter(*cloud_leaf);

        auto t1 = std::chrono::high_resolution_clock::now();
        double voxel_s = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0;

        std::cout << "\n========================\n";
        std::cout << "LEAF " << leaf << " -> " << cloud_leaf->size() << " pts (voxel " << voxel_s << " s)\n";
        std::cout << "========================\n";

        if (cloud_leaf->size() < 100)
        {
            std::cout << "Too few points after voxel. Skip leaf.\n";
            continue;
        }

        std::string downsample_file = out_dir + "/downsampled_leaf" + cmTag(leaf) + ".pcd";
        if (SAVE_DOWNSAMPLED)
        {
            std::cout << "[Save] " << downsample_file << "\n";
            writer.write<pcl::PointXYZ>(downsample_file, *cloud_leaf, true);
        }

        for (float dist_thresh : distance_thresholds)
        {
            case_id++;

            // --- controls ---
            const float MIN_REMOVAL_RATIO = 0.25f;  // 25%
            const int   MAX_GROUND_ITERS  = 10;     // safety cap
            const bool  SAVE_ITER_NONGROUND = false; // set true if you want per-iter outputs

            // Work on a copy, so other dist_thresh runs still start from same voxel cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr working(new pcl::PointCloud<pcl::PointXYZ>(*cloud_leaf));

            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(false);
            seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setMaxIterations(50);
            seg.setDistanceThreshold(dist_thresh);
            seg.setAxis(Eigen::Vector3f(0.f, 0.f, 1.f));
            seg.setEpsAngle(pcl::deg2rad(10.0f));

            std::size_t prev_removed = 0;
            std::size_t total_removed = 0;
            int it = 0;

            auto p0_total = std::chrono::high_resolution_clock::now();

            while (it < MAX_GROUND_ITERS && working->size() >= 100)
            {
                pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);

                auto p0 = std::chrono::high_resolution_clock::now();

                seg.setInputCloud(working);
                seg.segment(*inliers, *coeffs);

                const std::size_t removed_now = inliers->indices.size();

                // no plane -> stop
                if (removed_now == 0 || coeffs->values.size() < 4)
                {
                    std::cout << "Case " << case_id << " leaf=" << leaf << " dist=" << dist_thresh
                            << " | iter " << it << " -> NO MORE PLANES\n";
                    break;
                }

                // ratio test (only from 2nd iteration onwards)
                if (it > 0 && removed_now < static_cast<std::size_t>(MIN_REMOVAL_RATIO * prev_removed))
                {
                    std::cout << "Case " << case_id << " leaf=" << leaf << " dist=" << dist_thresh
                            << " | iter " << it
                            << " -> STOP (removed " << removed_now << " < "
                            << (MIN_REMOVAL_RATIO * prev_removed) << ")\n";
                    break;
                }

                // Remove inliers (ground) -> keep non-ground
                pcl::ExtractIndices<pcl::PointXYZ> extract;
                extract.setInputCloud(working);
                extract.setIndices(inliers);
                extract.setNegative(true);

                pcl::PointCloud<pcl::PointXYZ>::Ptr next(new pcl::PointCloud<pcl::PointXYZ>);
                extract.filter(*next);

                auto p1 = std::chrono::high_resolution_clock::now();
                double plane_iter_s =
                    std::chrono::duration_cast<std::chrono::milliseconds>(p1 - p0).count() / 1000.0;

                std::cout << "Case " << case_id
                        << " leaf=" << leaf << " dist=" << dist_thresh
                        << " | iter=" << it
                        << " removed=" << removed_now
                        << " remaining=" << next->size()
                        << " | " << plane_iter_s << " s\n";

                total_removed += removed_now;
                prev_removed = removed_now;
                working = next;
                ++it;

                // Optional: save intermediate nonground after each iteration
                if (SAVE_ITER_NONGROUND)
                {
                    std::string iter_file = out_dir + "/nonground_leaf" + cmTag(leaf)
                        + "_dist" + cmTag(dist_thresh) + "_iter" + std::to_string(it) + ".pcd";
                    writer.write<pcl::PointXYZ>(iter_file, *working, true);
                }
            }

            auto p1_total = std::chrono::high_resolution_clock::now();
            double plane_total_s =
                std::chrono::duration_cast<std::chrono::milliseconds>(p1_total - p0_total).count() / 1000.0;

            // Final output (after multiple removals)
            std::string nonground_file =
                out_dir + "/nonground_leaf" + cmTag(leaf) + "_dist" + cmTag(dist_thresh) + "_FINAL.pcd";

            if (SAVE_NONGROUND)
            {
                std::cout << "[Save] " << nonground_file << "\n";
                writer.write<pcl::PointXYZ>(nonground_file, *working, true);
            }

            // Log (summary per (leaf,dist))
            log << nowIso8601() << "," << case_id << ","
                << leaf << "," << dist_thresh << ","
                << cloud->size() << "," << cloud_leaf->size() << ","
                << total_removed << "," << working->size() << ","
                << voxel_s << "," << plane_total_s << ","
                << downsample_file << "," << nonground_file << "\n";
    log.flush();
        }
    }

    std::cout << "\nDone. Saved to: " << out_dir << "\n";
    std::cout << "Log: " << out_dir << "/ground_removal_log.csv\n";
    return 0;
}
