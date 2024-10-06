#include <iostream>
#include <vector>
#include <fstream>
#include <glob.h>
#include <sys/stat.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
int main() {
    // Directory where the files are
    std::string dataDirectory = "/mnt/carsedat/";
    std::string fileType;
    char choice;

    //if they want to process all files or a specific file
    std::cout << "Do you want to process all files or a specific file? (a for all, s for specific): ";
    std::cin >> choice;

    std::vector<std::string> files;

    if (choice == 'a') {
        // all files
        std::cout << "Enter the file format you want (e.g., .dat): ";
        std::cin >> fileType;
        std::string filePattern = dataDirectory + "*" + fileType;

        // Debugging output
        std::cout << "Searching for files with pattern: " << filePattern << std::endl;

        // Declare a struct to hold the results
        glob_t glob_result;
        int glob_status = glob(filePattern.c_str(), 0, NULL, &glob_result);
        if (glob_status != 0) {
            std::cerr << "glob() failed with return code " << glob_status << std::endl;
            switch (glob_status) {
                case GLOB_NOSPACE:
                    std::cerr << "GLOB_NOSPACE: Running out of memory." << std::endl;
                    break;
                case GLOB_ABORTED:
                    std::cerr << "GLOB_ABORTED: Read error." << std::endl;
                    break;
                case GLOB_NOMATCH:
                    std::cerr << "GLOB_NOMATCH: No matches found." << std::endl;
                    break;
                default:
                    std::cerr << "Unknown error." << std::endl;
                    break;
            }
            return EXIT_FAILURE;
        }

        // Iterate through the results and add all of the resultfiles to the vector
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            files.push_back(std::string(glob_result.gl_pathv[i]));
        }
        globfree(&glob_result);

        std::cout << "Found " << files.size() << " files." << std::endl;

    } else if (choice == 's') {
        // Process a specific file
        std::string fileName;
        std::cout << "Enter the filename: ";
        std::cin >> fileName;
        files.push_back(dataDirectory + fileName);
    } else {
        std::cerr << "Invalid choice." << std::endl;
        return EXIT_FAILURE;
    }

    int filecount = 0, packetcount = 0;

    // Initialize all the arrays to hold the data
    std::vector<int16_t> adc1(1024);
    std::vector<int16_t> adc2(1024);
    std::vector<int16_t> adc3(1024);
    std::vector<int16_t> adc4(1024);

    // Open the output file
    std::ofstream outputFile("carsedatoutput.txt");
    if (!outputFile) {
        std::cerr << "Error opening output file." << std::endl;
        return EXIT_FAILURE;
    }

    // Iterate through all of the files in the vector
    for (const std::string& filePath : files) {
        std::cout << "Processing file: " << filePath << std::endl;
        filecount++;
        std::ifstream file(filePath, std::ios::binary);
        if (!file) {
            std::cerr << "Error opening file: " << filePath << std::endl;
            continue;
        }

        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        size_t numPackets = file_size / 8256;

        // Initialize the vector to hold the 64-bit counter
        std::vector<uint64_t> Counter(numPackets);
	
        // Loop to cover the number of packets
        for (size_t i = 0; i < numPackets; i++) {
            packetcount++;
            uint64_t counter;
	fseek(file,56,SEEK_SET);


            file.read(reinterpret_cast<char*>(&counter), sizeof(counter));
            Counter[i] = counter;

            // Check if the counter was read successfully
            if (file.gcount() != sizeof(counter)) {
                std::cerr << "Error reading counter in file: " << filePath << " at packet: " << i << std::endl;
                break;
            }

            // Then we need a loop covering 1024 bytes which is the size of each packet part
            for (int j = 0; j < 1024; j++) {
                file.read(reinterpret_cast<char*>(&adc1[j]), sizeof(int16_t));
                file.read(reinterpret_cast<char*>(&adc2[j]), sizeof(int16_t));
                file.read(reinterpret_cast<char*>(&adc3[j]), sizeof(int16_t));
                file.read(reinterpret_cast<char*>(&adc4[j]), sizeof(int16_t));
            }


            // Write to the output file
            outputFile << "Counter: " << Counter[i] << "\n";
            outputFile << "adc1\tadc2\tadc3\tadc4\n";
            for (int j = 0; j < 1024; j++) {
                outputFile << adc1[j] << "\t" << adc2[j] << "\t" << adc3[j] << "\t" << adc4[j] << "\t\n";
            }

        }



file.close();
    }

    outputFile.close();
    return 0;
}

