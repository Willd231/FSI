#include <iostream>
#include <fstream>
#include <glob.h>
#include <sys/stat.h>
#include <vector>
#include <string>
#include <cstring>

// Function to check if a file exists
inline bool exists(const std::string &name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

int main() {
    // File specifics 
    std::string filetype;
    std::string Datdir = "/mnt/carsedat/";  
    std::string file_path;

    // Collect user input
    std::cout << "Enter the filename for the first file in the sequence: ";
    std::cin >> filetype;

    // Concatenate the path with directory
    file_path = Datdir + filetype;

    // Check if file path exists 
    if (!exists(file_path)) {
        std::cerr << file_path << " file does not exist!!" << std::endl;
        return 1;
    }

    // If we cannot open the file 
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    // Initialize the counter variable
    uint64_t b;
    file.read(reinterpret_cast<char*>(&b), sizeof(uint64_t));
    file.close();

    std::cout << b << std::endl;

    // Specify the file pattern to be collected by the glob function
    std::string pattern = Datdir + filetype.substr(0, filetype.length() - 8) + "*.dat";
    glob_t glob_result;

    // Call glob
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);

    // Initialize vector to hold files
    std::vector<std::string> files;

    // Put all of the glob array into the new vector
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        files.push_back(std::string(glob_result.gl_pathv[i]));
    }

    // Free memory taken up by glob array
    globfree(&glob_result);

    // Begin checking loop
    for (const std::string& file_path : files) {
        std::cout << file_path << std::endl;

        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            std::cerr << "Error opening file: " << file_path << std::endl;
            continue;
        }

        // Find file size to determine the number of packets
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Calculate the number of packets
        size_t total_num_packets = file_size / 8448;
        std::cout << "Total number of packets: " << total_num_packets << std::endl;

        size_t packet_cnt = 0;
        for (size_t j = 0; j < total_num_packets; ++j) {
            std::vector<uint64_t> intval(8448 / sizeof(uint64_t));
            file.read(reinterpret_cast<char*>(intval.data()), 8448);

            // Check the counter
            for (size_t k = 0; k < intval.size(); ++k) {
                uint64_t count_value = intval[k];
                if (count_value != b) {
                    std::cout << "Total Packet count: " << total_num_packets << std::endl;
                    std::cout << "Packet count: " << packet_cnt << std::endl;
                    std::cout << "Mismatched counter at: " << count_value << std::endl;
                    std::cout << "Expected counter: " << b << std::endl;
                    b = count_value + 1;
                } else {
                    b += 1;
                }
                ++packet_cnt;
            }
        }

        file.close();
    }

    return 0;
}
