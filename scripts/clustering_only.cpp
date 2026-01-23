#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/extract_indices.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdlib>

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

static bool file_exists(const std::string& p)
{
    struct stat st;
    return (stat(p.c_str(), &st) == 0) && S_ISREG(st.st_mode);
}

static std::time_t file_mtime(const std::string& p)
{
    struct stat st;
    if (stat(p.c_str(), &st) != 0) return 0;
    return st.st_mtime;
}

static long long file_size(const std::string& p)
{
    struct stat st;
    if (stat(p.c_str(), &st) != 0) return -1;
    return static_cast<long long>(st.st_size);
}

static std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static std::string get_ext(const std::string& p)
{
    size_t dot = p.find_last_of('.');
    if (dot == std::string::npos) return "";
    return to_lower(p.substr(dot));
}

static std::string replace_ext(const std::string& p, const std::string& newExt)
{
    size_t dot = p.find_last_of('.');
    if (dot == std::string::npos) return p + newExt;
    return p.substr(0, dot) + newExt;
}

// Convert LAS/LAZ to PCD, ensuring Red + Infrared fields are present
static bool ensure_pcd_from_las(const std::string& lasPath, const std::string& pcdPath)
{
    bool needConvert = true;

    if (file_exists(pcdPath))
    {
        // Don’t trust only mtime; also require non-trivial size
        if (file_mtime(pcdPath) >= file_mtime(lasPath) && file_size(pcdPath) > 200)
            needConvert = false;
    }

    if (!needConvert) return true;

    std::ostringstream cmd;
    cmd << "pdal translate "
    << "\"" << lasPath << "\" "
    << "\"" << pcdPath << "\" "
    << "--writers.pcd.compression=binary "
    << "--writers.pcd.order=\"X=Float,Y=Float,Z=Float,Red=Unsigned16,Infrared=Unsigned16\" "
    << "--writers.pcd.keep_unspecified=true";


    std::cout << "Converting LAS/LAZ -> PCD using PDAL:\n  " << cmd.str() << "\n";
    int ret = std::system(cmd.str().c_str());
    if (ret != 0 || !file_exists(pcdPath) || file_size(pcdPath) <= 200)
    {
        std::cerr << "PDAL conversion failed or produced empty PCD.\n";
        return false;
    }
    return true;
}

static bool loadPCDWithAttrs(const std::string& filename,
                            pcl::PCLPointCloud2::Ptr cloud_full,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz,
                            std::string& out_pcd_path)
{
    if (!file_exists(filename))
    {
        std::cerr << "Input file does not exist: " << filename << "\n";
        return false;
    }

    const std::string ext = get_ext(filename);
    std::string pcdPath;

    if (ext == ".pcd")
    {
        pcdPath = filename;
    }
    else if (ext == ".las" || ext == ".laz")
    {
        pcdPath = replace_ext(filename, ".pcd");
        if (!ensure_pcd_from_las(filename, pcdPath))
            return false;
    }
    else
    {
        std::cerr << "Unsupported extension: " << ext << " (expected .pcd/.las/.laz)\n";
        return false;
    }

    pcl::PCDReader reader;
    if (reader.read(pcdPath, *cloud_full) < 0)
    {
        std::cerr << "Error reading PCD: " << pcdPath << "\n";
        return false;
    }

    pcl::fromPCLPointCloud2(*cloud_full, *cloud_xyz);

    out_pcd_path = pcdPath;
    std::cout << "Loaded PCD(full fields): " << pcdPath
              << " (pts=" << cloud_xyz->size() << ")\n";

    // Helpful: print available fields
    std::cout << "Fields in PCD: ";
    for (const auto& f : cloud_full->fields) std::cout << f.name << " ";
    std::cout << "\n";

    return true;
}

int main()
{
    const std::string out_dir = "out_cluster";
    ensure_dir(out_dir);

    const std::string input_nonground =
        "./out_ground/2025-07-15-MS_Vinograd_1_classified_smrf_non_ground.las";

    const bool SAVE_CLUSTERS = true;

    std::vector<float> leaf_sizes         = {0.00f};  // keep 0 for now
    std::vector<float> cluster_tolerances = {0.4f};

    const int MIN_CLUSTER_SIZE = 50000;
    const int MAX_CLUSTER_SIZE = 850000;

    std::ofstream log(out_dir + "/clustering_only_log.csv", std::ios::out);
    log << "timestamp,config_id,leaf_m,tol_m,input_pts,clusters,valid,reason,input_file\n";

    pcl::PCLPointCloud2::Ptr input_full(new pcl::PCLPointCloud2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    std::string input_pcd_path;

    if (!loadPCDWithAttrs(input_nonground, input_full, input_xyz, input_pcd_path))
        return -1;

    int config_id = 0;
    const auto global_start = std::chrono::high_resolution_clock::now();

    for (float leaf : leaf_sizes)
    {
        // NOTE: leaf>0 voxel downsample not implemented for full cloud here.
        // Keep leaf=0 to preserve 1:1 indexing between full and xyz.
        if (leaf > 0.0f)
        {
            std::cerr << "leaf>0 not supported in this version (would break attribute indexing). Use leaf=0.\n";
            continue;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_work_xyz(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PCLPointCloud2::Ptr cloud_work_full(new pcl::PCLPointCloud2);

        *cloud_work_xyz  = *input_xyz;
        *cloud_work_full = *input_full;

        if (cloud_work_xyz->size() < 100)
        {
            std::cout << "Too few points, skip.\n";
            continue;
        }

        // KD-tree on XYZ
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_work_xyz);

        for (float tol : cluster_tolerances)
        {
            config_id++;

            std::cout << "\nCONFIG " << config_id
                      << " | leaf=" << leaf
                      << " | tol=" << tol << "\n";

            auto c0 = std::chrono::high_resolution_clock::now();

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(tol);
            ec.setMinClusterSize(MIN_CLUSTER_SIZE);
            ec.setMaxClusterSize(MAX_CLUSTER_SIZE);
            ec.setSearchMethod(tree);
            ec.setInputCloud(cloud_work_xyz);
            ec.extract(cluster_indices);

            auto c1 = std::chrono::high_resolution_clock::now();
            double cluster_s =
                std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count() / 1000.0;

            const bool valid = (cluster_indices.size() <= 50);
            const std::string reason = valid ? "ok" : "too_many_clusters";

            std::cout << "clusters=" << cluster_indices.size()
                      << " | time=" << cluster_s << " s"
                      << (valid ? " | VALID\n" : " | INVALID\n");

            if (SAVE_CLUSTERS && valid)
            {
                pcl::PCDWriter writer2;
                int j = 0;

                for (const auto& cl : cluster_indices)
                {
                    pcl::PointIndices::Ptr inds(new pcl::PointIndices(cl));

                    pcl::PCLPointCloud2 cluster_full;
                    pcl::ExtractIndices<pcl::PCLPointCloud2> ex;
                    ex.setInputCloud(cloud_work_full);
                    ex.setIndices(inds);
                    ex.setNegative(false);
                    ex.filter(cluster_full);

                    std::ostringstream ss;
                    ss << out_dir << "/config" << config_id
                       << "_leaf" << cmTag(leaf)
                       << "_tol" << cmTag(tol)
                       << "_cluster_" << std::setw(2) << std::setfill('0') << j << ".pcd";

                    writer2.writeBinary(ss.str(), cluster_full);
                    ++j;
                }
            }

            log << nowIso8601() << "," << config_id << ","
                << leaf << "," << tol << ","
                << cloud_work_xyz->size() << ","
                << cluster_indices.size() << ","
                << (valid ? 1 : 0) << ","
                << reason << ","
                << input_pcd_path << "\n";
            log.flush();
        }
    }

    auto now = std::chrono::high_resolution_clock::now();
    double total_s =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - global_start).count() / 1000.0;
    std::cout << "\nDone in " << total_s << " s. Log: " << out_dir + "/clustering_only_log.csv" << "\n";
    return 0;
}
