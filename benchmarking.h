#ifndef CCBS_BENCHMARKING_H_
#define CCBS_BENCHMARKING_H_

#include <string>
#include <vector>

// Forward declarations
class Solution;

struct BenchmarkJob {
  std::string map_file;
  std::string task_file;
  std::string original_config_file;
  int connectedness;
  std::string task_name;
  int job_id;
};

struct BenchmarkResult {
  std::string task_name;
  int connectedness;
  int num_agents;
  bool solved;
  double runtime;
  double makespan;
  double flowtime;
  double init_cost;
  double check_time;
  int high_level_expanded;
  int low_level_expansions;
  double low_level_expanded;
};

class MultiCoreBenchmarkRunner {
 public:
  explicit MultiCoreBenchmarkRunner(int max_processes = 12);
  
  void run_benchmark_suite(const std::string& folder_path,
                          int min_connectedness,
                          int max_connectedness,
                          const std::string& output_csv = "benchmark_results.csv");

 private:
  int max_processes_;
  std::string temp_dir_;
  
  std::vector<BenchmarkJob> generate_all_jobs(const std::string& folder_path,
                                              int min_connectedness,
                                              int max_connectedness);
  
  std::vector<std::string> find_task_files(const std::string& folder_path);
  
  void process_job_queue_multicore(const std::vector<BenchmarkJob>& jobs);
  
  std::vector<BenchmarkResult> process_job(const BenchmarkJob& job);
  
  std::string create_temp_config_with_connectedness(const BenchmarkJob& job);
  
  int count_agents_in_task(const std::string& task_file);
  
  std::string create_subset_task_file(const std::string& original_task_file,
                                     int num_agents,
                                     int job_id);
  
  Solution run_ccbs_single(const std::string& map_file,
                          const std::string& task_file,
                          const std::string& config_file);
  
  void write_results_to_temp_file(const std::vector<BenchmarkResult>& results,
                                 int job_id);
  
  void collect_and_write_final_results(const std::string& output_csv);
  
  void cleanup_temp_directory();
};

/**
 * Main benchmarking function that runs CCBS on different connectedness values
 * and agent counts for all task files in the given folder.
 * 
 * @param folder_path Path to folder containing map.xml, config.xml, and task files
 * @param min_connectedness Minimum connectedness value to test
 * @param max_connectedness Maximum connectedness value to test
 * @param num_cores Number of CPU cores to use for parallel processing
 * @param output_file Output CSV filename for results
 */
void run_benchmark(const std::string& folder_path,
                  int min_connectedness,
                  int max_connectedness,
                  int num_cores = 12,
                  const std::string& output_file = "benchmark_results.csv");

#endif  // CCBS_BENCHMARKING_H_
