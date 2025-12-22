#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/angles.h>

#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/Options.hpp>
#include <pdal/io/LasReader.hpp>
#include <pdal/io/LasHeader.hpp>

#include <Eigen/Core>

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

using PointT = pcl::PointXYZRGB;

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

static inline uint8_t u16_to_u8(uint16_t v)
{
    // LAS RGB često dolazi 0..65535; mapiraj u 0..255
    // (ovo je brže od float skaliranja)
    return static_cast<uint8_t>(v >> 8); // v/256
}

// ---------- RGB -> HSV (0..360, 0..1, 0..1) ----------
static void rgbToHsv(uint8_t r8, uint8_t g8, uint8_t b8, float& h, float& s, float& v)
{
    float r = r8 / 255.0f;
    float g = g8 / 255.0f;
    float b = b8 / 255.0f;

    float cmax = std::max({r, g, b});
    float cmin = std::min({r, g, b});
    float delta = cmax - cmin;

    v = cmax;

    if (cmax <= 1e-6f) { s = 0.0f; h = 0.0f; return; }
    s = (delta <= 1e-6f) ? 0.0f : (delta / cmax);

    if (delta <= 1e-6f) { h = 0.0f; return; }

    if (cmax == r)      h = 60.0f * std::fmod(((g - b) / delta), 6.0f);
    else if (cmax == g) h = 60.0f * (((b - r) / delta) + 2.0f);
    else                h = 60.0f * (((r - g) / delta) + 4.0f);

    if (h < 0.0f) h += 360.0f;
}

// ---------- Ground-like color classifier ----------
// Ideja: ground često ~ smeđa/oranž + tamne nijanse (crno/sivo) zbog sjena.
// Ovo je namjerno "šire" da ne propustiš zemlju.
static bool isGroundLikeColor(uint8_t r, uint8_t g, uint8_t b)
{
    float h, s, v;
    rgbToHsv(r, g, b, h, s, v);

    // 1) tamno / crno / sjene (neovisno o hue)
    //    - niska vrijednost (brightness)
    if (v < 0.18f) return true;

    // 2) tamno sivo (nizak saturation, ali još uvijek dosta tamno)
    if (v < 0.30f && s < 0.20f) return true;

    // 3) smeđe / narančasto / tamno-narančasto / "earth tones"
    //    Hue oko 10°..55° (crvenkasto->žuto/narančasto), solidna saturacija.
    if (h >= 10.0f && h <= 55.0f && s >= 0.20f && v >= 0.15f) return true;

    // 4) "tan / burlywood" znaju biti svjetliji (visok v), ali opet u tom hue području
    if (h >= 15.0f && h <= 65.0f && s >= 0.10f && v >= 0.55f) return true;

    return false;
}

// ---------- IO: PCD or LAS ----------
bool loadPointCloud(const std::string& filename, pcl::PointCloud<PointT>::Ptr cloud)
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
            std::cout << "  Median X,Y,Z (sampled): ("
                      << medianOf(xs) << ", " << medianOf(ys) << ", " << medianOf(zs) << ")\n";

            // Copy into PCL cloud (XYZ + RGB)
            cloud->clear();
            cloud->points.reserve(view->size());

            std::size_t withColor = 0;

            for (pdal::PointId i = 0; i < view->size(); ++i)
            {
                PointT p;
                p.x = static_cast<float>(view->getFieldAs<double>(pdal::Dimension::Id::X, i));
                p.y = static_cast<float>(view->getFieldAs<double>(pdal::Dimension::Id::Y, i));
                p.z = static_cast<float>(view->getFieldAs<double>(pdal::Dimension::Id::Z, i));

                // LAS RGB are typically uint16
                // If the LAS lacks RGB dimensions, PDAL will throw -> caught by try/catch
                uint16_t r16 = view->getFieldAs<uint16_t>(pdal::Dimension::Id::Red, i);
                uint16_t g16 = view->getFieldAs<uint16_t>(pdal::Dimension::Id::Green, i);
                uint16_t b16 = view->getFieldAs<uint16_t>(pdal::Dimension::Id::Blue, i);

                p.r = u16_to_u8(r16);
                p.g = u16_to_u8(g16);
                p.b = u16_to_u8(b16);
                ++withColor;

                cloud->points.push_back(p);
            }

            cloud->width = static_cast<uint32_t>(cloud->size());
            cloud->height = 1;
            cloud->is_dense = true;

            std::cout << "Loaded into PCL: " << cloud->size()
                      << " pts | RGB read: " << withColor << "\n";
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
    const bool SAVE_ORIGINAL         = false; // original 18M je ogroman
    const bool SAVE_DOWNSAMPLED      = true;
    const bool SAVE_NONGROUND_FINAL  = true;
    const bool SAVE_GROUND_FINAL     = false;
    const bool SAVE_ITER_NONGROUND   = false; // true ako želiš vidjeti progres po iteracijama

    // ---------- input ----------
    const std::string input_file = "../datasource/LOCAL_VINEYARD_MS.las";

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    if (!loadPointCloud(input_file, cloud))
    {
        std::cerr << "Failed to load: " << input_file << "\n";
        return -1;
    }

    pcl::PCDWriter writer;

    if (SAVE_ORIGINAL)
    {
        const std::string orig_path = out_dir + "/original_cloud_rgb.pcd";
        std::cout << "[Save] " << orig_path << "\n";
        writer.write<PointT>(orig_path, *cloud, true);
    }

    // ---------- sweep params (meters) ----------
    std::vector<float> leaf_sizes          = {0.19f, 0.20f, 0.22f};
    std::vector<float> distance_thresholds = {0.02f, 0.08f, 0.12f};

    // Iterative ground removal controls
    const float MIN_REMOVAL_RATIO = 0.25f;  // 25% od prethodnog uklanjanja
    const int   MAX_GROUND_ITERS  = 10;     // sigurnosna kapa
    const int   MIN_CANDIDATES    = 5000;   // ako je premalo "ground-like" točaka, nema smisla fitati

    // Log per (leaf,dist)
    std::ofstream log(out_dir + "/ground_removal_log.csv", std::ios::out);
    log << "timestamp,case_id,leaf_m,dist_m,orig_pts,voxel_pts,"
           "candidates0,total_removed,final_nonground,"
           "voxel_s,plane_s,"
           "downsample_file,nonground_file\n";

    int case_id = 0;

    for (float leaf : leaf_sizes)
    {
        // Voxel once per leaf
        auto t0 = std::chrono::high_resolution_clock::now();

        pcl::VoxelGrid<PointT> vg;
        pcl::PointCloud<PointT>::Ptr cloud_leaf(new pcl::PointCloud<PointT>);
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
            writer.write<PointT>(downsample_file, *cloud_leaf, true);
        }

        for (float dist_thresh : distance_thresholds)
        {
            case_id++;

            // work on a copy so each dist run starts from same voxel output
            pcl::PointCloud<PointT>::Ptr working(new pcl::PointCloud<PointT>(*cloud_leaf));

            pcl::SACSegmentation<PointT> seg;
            seg.setOptimizeCoefficients(false);
            seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setMaxIterations(50);
            seg.setDistanceThreshold(dist_thresh);
            seg.setAxis(Eigen::Vector3f(0.f, 0.f, 1.f));
            seg.setEpsAngle(pcl::deg2rad(15.0f)); // malo šire za nagibe (po želji vrati na 10)

            std::size_t prev_removed = 0;
            std::size_t total_removed = 0;
            int it = 0;

            auto p0_total = std::chrono::high_resolution_clock::now();

            std::size_t candidates0 = 0;

            while (it < MAX_GROUND_ITERS && working->size() >= 100)
            {
                // ---------- COLOR PREFILTER: build candidate indices ----------
                pcl::PointIndices::Ptr candidates(new pcl::PointIndices);
                candidates->indices.reserve(working->size() / 3);

                for (int i = 0; i < static_cast<int>(working->size()); ++i)
                {
                    const auto& p = (*working)[i];
                    if (isGroundLikeColor(p.r, p.g, p.b))
                        candidates->indices.push_back(i);
                }

                if (it == 0) candidates0 = candidates->indices.size();

                if (static_cast<int>(candidates->indices.size()) < MIN_CANDIDATES)
                {
                    std::cout << "Case " << case_id
                              << " leaf=" << leaf << " dist=" << dist_thresh
                              << " | iter " << it
                              << " -> STOP (too few color candidates: " << candidates->indices.size() << ")\n";
                    break;
                }

                pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);

                auto p0 = std::chrono::high_resolution_clock::now();

                seg.setInputCloud(working);
                seg.setIndices(candidates);          // <<< KEY: RANSAC only on "ground-like" colors
                seg.segment(*inliers, *coeffs);

                auto p1 = std::chrono::high_resolution_clock::now();

                const std::size_t removed_now = inliers->indices.size();

                // no plane -> stop
                if (removed_now == 0 || coeffs->values.size() < 4)
                {
                    std::cout << "Case " << case_id
                              << " leaf=" << leaf << " dist=" << dist_thresh
                              << " | iter " << it << " -> NO MORE PLANES\n";
                    break;
                }

                // ratio test from 2nd iteration onwards
                if (it > 0 && removed_now < static_cast<std::size_t>(MIN_REMOVAL_RATIO * prev_removed))
                {
                    std::cout << "Case " << case_id
                              << " leaf=" << leaf << " dist=" << dist_thresh
                              << " | iter " << it
                              << " -> STOP (removed " << removed_now
                              << " < " << (MIN_REMOVAL_RATIO * prev_removed) << ")\n";
                    break;
                }

                // Remove inliers (ground) -> keep non-ground
                pcl::ExtractIndices<PointT> extract;
                extract.setInputCloud(working);
                extract.setIndices(inliers);
                extract.setNegative(true); // keep non-ground

                pcl::PointCloud<PointT>::Ptr next(new pcl::PointCloud<PointT>);
                extract.filter(*next);

                double iter_s = std::chrono::duration_cast<std::chrono::milliseconds>(p1 - p0).count() / 1000.0;

                std::cout << "Case " << case_id
                          << " leaf=" << leaf << " dist=" << dist_thresh
                          << " | iter=" << it
                          << " candidates=" << candidates->indices.size()
                          << " removed=" << removed_now
                          << " remaining=" << next->size()
                          << " | " << iter_s << " s\n";

                total_removed += removed_now;
                prev_removed = removed_now;
                working = next;
                ++it;

                if (SAVE_ITER_NONGROUND)
                {
                    std::string iter_file = out_dir + "/nonground_leaf" + cmTag(leaf)
                        + "_dist" + cmTag(dist_thresh) + "_iter" + std::to_string(it) + ".pcd";
                    writer.write<PointT>(iter_file, *working, true);
                }
            }

            auto p1_total = std::chrono::high_resolution_clock::now();
            double plane_total_s =
                std::chrono::duration_cast<std::chrono::milliseconds>(p1_total - p0_total).count() / 1000.0;

            // Final output
            std::string nonground_file =
                out_dir + "/nonground_leaf" + cmTag(leaf) + "_dist" + cmTag(dist_thresh) + "_FINAL.pcd";

            if (SAVE_NONGROUND_FINAL)
            {
                std::cout << "[Save] " << nonground_file << "\n";
                writer.write<PointT>(nonground_file, *working, true);
            }

            if (SAVE_GROUND_FINAL)
            {
                // Ovdje više nemamo "ground cloud" eksplicitno jer ga iterativno mičemo.
                // Ako treba: može se spremiti kao (cloud_leaf - working) preko ExtractIndices na union indeksa.
                // Za sada ostavljam isključeno da ne kompliciram.
            }

            log << nowIso8601() << "," << case_id << ","
                << leaf << "," << dist_thresh << ","
                << cloud->size() << "," << cloud_leaf->size() << ","
                << candidates0 << ","
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
