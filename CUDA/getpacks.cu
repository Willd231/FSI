// file: collect_packets.cu
#include <cstdio>
#include <cstdint>
#include <vector>
#include <array>
#include <iostream>

struct Packet {
    uint64_t counter[8];
    int16_t  data[1024];
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    const char* filename = argv[1];
    FILE* fp = std::fopen(filename, "rb");
    if (!fp) {
        std::perror("fopen");
        return 1;
    }

    // Storage for all packets' counters
    std::vector< std::array<uint64_t,8> > all_counters;
    // Four ADC vectors
    std::vector<int16_t> adc1, adc2, adc3, adc4;

    // How many packets to read per batch
    const size_t PACKETS_PER_BATCH = 128;
    all_counters.reserve(10000);
    adc1.reserve(10000 * 256);
    adc2.reserve(10000 * 256);
    adc3.reserve(10000 * 256);
    adc4.reserve(10000 * 256);

    while (true) {
        // read a batch of packets
        std::vector<Packet> batch;
        batch.reserve(PACKETS_PER_BATCH);
        for (size_t i = 0; i < PACKETS_PER_BATCH; ++i) {
            Packet p;
            if (std::fread(&p, sizeof(Packet), 1, fp) != 1) {
                break;
            }
            batch.push_back(p);
        }
        if (batch.empty()) break;

        // process batch
        for (auto &pkt : batch) {
            // store counters
            std::array<uint64_t,8> ctrs;
            for (int j = 0; j < 8; ++j) {
                ctrs[j] = pkt.counter[j];
            }
            all_counters.push_back(ctrs);

            // de-interleave 1024 samples into 4 ADCs of 256 samples each
            for (size_t s = 0; s < 256; ++s) {
                adc1.push_back(pkt.data[4*s + 0]);
                adc2.push_back(pkt.data[4*s + 1]);
                adc3.push_back(pkt.data[4*s + 2]);
                adc4.push_back(pkt.data[4*s + 3]);
            }
        }
    }

    std::fclose(fp);

    std::cout << "Total packets read: " << all_counters.size() << "\n";
    std::cout << "ADC1 samples: " << adc1.size() << "\n";
    std::cout << "ADC2 samples: " << adc2.size() << "\n";
    std::cout << "ADC3 samples: " << adc3.size() << "\n";
    std::cout << "ADC4 samples: " << adc4.size() << "\n";

    return 0;
}
