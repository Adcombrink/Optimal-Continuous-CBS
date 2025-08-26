#include <cstdlib>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <dirent.h>
#include <sys/stat.h>
#include "tinyxml2.h"
#include "map.h"
#include "task.h"
#include "cbs.h"
#include "config.h"

using namespace tinyxml2;

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
    private:
        int max_processes_;
        std::string temp_dir_;
  
    public:
    explicit MultiCoreBenchmarkRunner(int max_processes = 12) : max_processes_(max_processes) {

    }
  
    void run_benchmark_suite(const std::string& folder_path, int min_connectedness, int max_connectedness,const std::string& output_csv = "benchmark_results.csv") {
        /* Runs the benchmark suite for a given problem and outputs the results to a CSV file.

        One benchmark is defined by a map, config, task, and a connectedness value: solve for 2 agents, then 
        increase the number of agents by 1 and repeat, until no solution is found before timeout.
        */

        // Create temp directory in the same folder as input files
        temp_dir_ = folder_path + "/ccbs_temp";
        mkdir(temp_dir_.c_str(), 0755);
        std::cout << "Temporary directory created: " << temp_dir_ << std::endl;
        
        auto jobs = generate_all_jobs(folder_path, min_connectedness, max_connectedness);
        std::cout << "Generated " << jobs.size() << " jobs" << std::endl;
        
        process_job_queue_multicore(jobs);
        collect_and_write_final_results(output_csv);
        cleanup_temp_directory();
    }
  
    private:
    std::vector<BenchmarkJob> generate_all_jobs(const std::string& folder_path, int min_connectedness,  int max_connectedness) {
        /* Generates all benchmark jobs for a given folder and connectedness range. */

        std::vector<BenchmarkJob> jobs;
        int job_counter = 0;
        
        auto task_files = find_task_files(folder_path);
        std::string map_file = folder_path + "/map.xml";
        std::string config_file = folder_path + "/config.xml";
        
        for (int conn = min_connectedness; conn <= max_connectedness; conn++) {
            for (const auto& task_file : task_files) {
                BenchmarkJob job;
                job.map_file = map_file;
                job.task_file = task_file;
                job.original_config_file = config_file;
                job.connectedness = conn;
                
                // Extract filename from path
                size_t pos = task_file.find_last_of("/");
                job.task_name = (pos == std::string::npos) ? task_file : task_file.substr(pos + 1);
                
                job.job_id = job_counter++;
                jobs.push_back(job);

                // std::cout << "Generated job " << job.job_id << ": " << job.task_name << " conn=" << job.connectedness << std::endl;

            }
        }

        return jobs;
    }
  
    std::vector<std::string> find_task_files(const std::string& folder_path) {
        std::vector<std::string> task_files;
        
        DIR* dir = opendir(folder_path.c_str());
        if (!dir) return task_files;
        
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            if (filename.length() > 4 && 
                filename.substr(filename.length() - 4) == ".xml" &&
                filename != "map.xml" && 
                filename != "config.xml") {
                task_files.push_back(folder_path + "/" + filename);
            }
        }
        
        closedir(dir);
        return task_files;
    }
  
    void process_job_queue_multicore(const std::vector<BenchmarkJob>& jobs) {
        std::vector<pid_t> active_processes;
        std::queue<BenchmarkJob> pending_jobs;
        
        for (const auto& job : jobs) {
            pending_jobs.push(job);
        }
        
        while (!pending_jobs.empty() || !active_processes.empty()) {
            while ((int)active_processes.size() < max_processes_ && !pending_jobs.empty()) {
                
                auto job = pending_jobs.front();
                pending_jobs.pop();
                
                pid_t pid = fork();
                if (pid == 0) {
                    auto results = process_job(job);
                    write_results_to_temp_file(results, job.job_id);
                    exit(0);
                } else if (pid > 0) {
                    active_processes.push_back(pid);
                    std::cout << "Started job " << job.job_id << ": " << job.task_name << " k=" << job.connectedness << std::endl;
                } else {
                    std::cerr << "Fork failed for job " << job.job_id << std::endl;
                }

            }
            
            int status;
            pid_t completed_pid = wait(&status);
            active_processes.erase(std::remove(active_processes.begin(), active_processes.end(), completed_pid), active_processes.end());
        }
    }
  
    std::vector<BenchmarkResult> process_job(const BenchmarkJob& job) {
        
        std::vector<BenchmarkResult> results;
        
        try {
            auto temp_config = create_temp_config_with_connectedness(job);
            int total_agents = count_agents_in_task(job.task_file);
            std::cout << "    Job " << job.job_id << ": " << job.task_name << " conn=" << job.connectedness << " with up to " << total_agents << " agents." << std::endl;

            for (int num_agents = 2; num_agents <= total_agents; num_agents++) {

                std::cout << "    Solving for " << num_agents << " agents..." << std::endl;
                auto temp_task = create_subset_task_file(job.task_file, num_agents, job.job_id);

                // run CCBS
                auto ccbs_result = run_ccbs_single(job.map_file, temp_task, temp_config);

                std::cout << "    Result for " << num_agents << " agents: "
                          << (ccbs_result.found ? "Solved" : "Unsolved") 
                          << " in " << ccbs_result.time.count() << " seconds." << std::endl;

                BenchmarkResult result;
                result.task_name = job.task_name;
                result.connectedness = job.connectedness;
                result.num_agents = num_agents;
                result.solved = ccbs_result.found;
                result.runtime = ccbs_result.time.count();
                result.makespan = ccbs_result.makespan;
                result.flowtime = ccbs_result.flowtime;
                result.init_cost = ccbs_result.init_cost;
                result.check_time = ccbs_result.check_time;
                result.high_level_expanded = ccbs_result.high_level_expanded;
                result.low_level_expansions = ccbs_result.low_level_expansions;
                result.low_level_expanded = ccbs_result.low_level_expanded;
                
                results.push_back(result);
                unlink(temp_task.c_str());  // Remove temp task file
                
                if (!ccbs_result.found) {
                    std::cout << "    ------ Stopping further tests for this job due to unsolved instance with number of agents: " << num_agents << std::endl;
                    break;
                }
            }
            
            unlink(temp_config.c_str());  // Remove temp config file

        } catch (const std::exception& e) {
            std::cerr << "Error processing job " << job.job_id << ": " << e.what() << std::endl;
        }
        
        return results;
    }
    
    std::string create_temp_config_with_connectedness(const BenchmarkJob& job) {
        std::string filename = "config_" + std::to_string(job.job_id) + ".xml";
        std::string temp_path = temp_dir_ + "/" + filename;
        
        XMLDocument doc;
        if (doc.LoadFile(job.original_config_file.c_str()) != XML_SUCCESS) {
            throw std::runtime_error("Failed to load config file");
        }
        
        auto root = doc.FirstChildElement("root");
        if (!root) {
            throw std::runtime_error("No root element in config file");
        }
        
        // Find the algorithm element first
        auto algorithm_elem = root->FirstChildElement("algorithm");
        if (!algorithm_elem) {
            throw std::runtime_error("No algorithm element in config file");
        }
        
        // Look for connectedness under algorithm, not root
        auto connectedness_elem = algorithm_elem->FirstChildElement("connectedness");
        if (connectedness_elem) {
            connectedness_elem->SetText(job.connectedness);
        } else {
            // If it doesn't exist under algorithm, create it there
            auto new_elem = doc.NewElement("connectedness");
            new_elem->SetText(job.connectedness);
            algorithm_elem->InsertEndChild(new_elem);
        }
        
        if (doc.SaveFile(temp_path.c_str()) != XML_SUCCESS) {
            throw std::runtime_error("Failed to save temp config file");
        }
        
        return temp_path;
    }

    int count_agents_in_task(const std::string& task_file) {
        XMLDocument doc;
        if (doc.LoadFile(task_file.c_str()) != XML_SUCCESS) {
            throw std::runtime_error("Failed to load task file");
        }
        
        auto root = doc.FirstChildElement("root");
        if (!root) {
            throw std::runtime_error("Missing root element in task file");
        }
        
        // Count <agent> elements directly under <root>
        int count = 0;
        for (auto agent = root->FirstChildElement("agent"); 
            agent; 
            agent = agent->NextSiblingElement("agent")) {
            count++;
        }
        
        return count;
    }
    
    std::string create_subset_task_file(const std::string& original_task_file, int num_agents, int job_id) {
        /* 
        Create a subset task file with a limited number of agents 
        */
        std::string filename = "task_" + std::to_string(job_id) + "_" + std::to_string(num_agents) + ".xml";
        std::string temp_path = temp_dir_ + "/" + filename;

        XMLDocument doc;
        if (doc.LoadFile(original_task_file.c_str()) != XML_SUCCESS) {
            throw std::runtime_error("Failed to load original task file");
        }
        
        auto root = doc.FirstChildElement("root");
        if (!root) {
            throw std::runtime_error("Missing root element in task file");
        }
        
        // Remove excess agents (keep only first num_agents)
        int count = 0;
        auto agent = root->FirstChildElement("agent");
        while (agent) {
            auto next_agent = agent->NextSiblingElement("agent");
            if (count >= num_agents) {
                root->DeleteChild(agent);
            }
            count++;
            agent = next_agent;
        }
        
        if (doc.SaveFile(temp_path.c_str()) != XML_SUCCESS) {
            throw std::runtime_error("Failed to save temp task file");
        }
        
        return temp_path;
    }

    Solution run_ccbs_single(const std::string& map_file, const std::string& task_file, const std::string& config_file) {

        Config config;
        config.getConfig(config_file.c_str());

        Map map(config.agent_size, config.connectdness);
        map.get_map(map_file.c_str());

        Task task;
        task.get_task(task_file.c_str());
        
        if (map.is_roadmap()) {
            task.make_ij(map);
        } else {
            task.make_ids(map.get_width());
        }
        
        CBS cbs;
        return cbs.find_solution(map, task, config);
    }
    
    void write_results_to_temp_file(const std::vector<BenchmarkResult>& results,
                                    int job_id) {
        std::string filename = "results_" + std::to_string(job_id) + ".csv";
        std::string temp_file = temp_dir_ + "/" + filename;
        
        std::ofstream file(temp_file.c_str());
        
        for (const auto& result : results) {
            file << result.task_name << ","
                << result.connectedness << ","
                << result.num_agents << ","
                << (result.solved ? "true" : "false") << ","
                << result.runtime << ","
                << result.makespan << ","
                << result.flowtime << ","
                << result.init_cost << ","
                << result.check_time << ","
                << result.high_level_expanded << ","
                << result.low_level_expansions << ","
                << result.low_level_expanded << std::endl;
        }
    }
    
    void collect_and_write_final_results(const std::string& output_csv) {
        std::ofstream final_file(output_csv.c_str());
        final_file << "task_name,connectedness,num_agents,solved,runtime,makespan,"
                << "flowtime,init_cost,check_time,hl_expanded,ll_searches,ll_expanded_avg"
                << std::endl;
        
        DIR* dir = opendir(temp_dir_.c_str());
        if (!dir) return;
        
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
            if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".csv") {
                std::string filepath = temp_dir_ + "/" + filename;
                std::ifstream temp_file(filepath.c_str());
                std::string line;
                while (std::getline(temp_file, line)) {
                    final_file << line << std::endl;
                }
            }
        }
        
        closedir(dir);
            std::cout << "Results written to: " << output_csv << std::endl;
    }
    
    void cleanup_temp_directory() {
        std::string command = "rm -rf " + temp_dir_;
        system(command.c_str());
    }
};

void run_benchmark(const std::string& folder_path, int min_connectedness, int max_connectedness, int num_cores = 12, const std::string& output_file = "benchmark_results.csv") {
    
    // Check if files exist
    std::ifstream map_check((folder_path + "/map.xml").c_str());
    std::ifstream config_check((folder_path + "/config.xml").c_str());
    if (!map_check.good() || !config_check.good()) {
        std::cerr << "Folder must contain map.xml and config.xml" << std::endl;
        return;
    }

    // Create full output path inside the given folder
    std::string full_output_path = output_file;
    if (!output_file.empty() && output_file[0] != '/') {  // Not an absolute path
        full_output_path = folder_path + "/" + output_file;
    }
    
    MultiCoreBenchmarkRunner runner(num_cores);
    runner.run_benchmark_suite(folder_path, min_connectedness, max_connectedness, full_output_path);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] 
                << " <folder_path> <min_connectedness> <max_connectedness> [num_cores] [output_file]"
                << std::endl;
        return 1;
    }
    
    std::string folder_path = argv[1];
    int min_connectedness = std::stoi(argv[2]);
    int max_connectedness = std::stoi(argv[3]);
    int num_cores = (argc > 4) ? std::stoi(argv[4]) : 12;
    std::string output_file = (argc > 5) ? argv[5] : "benchmark_results.csv";
    
    run_benchmark(folder_path, min_connectedness, max_connectedness, num_cores, output_file);
    
    return 0;
}
