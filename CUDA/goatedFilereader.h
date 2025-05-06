
#ifndef goatedFilereader_H
#define goatedFilereader_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <vector> 

using namespace std; 


typedef struct packet{
//int16_t * adc1, *adc2, *adc3, *adc4;
    std::vector<int16_t> adc1, adc2, adc3, adc4;
} packet;


inline std::vector<packet> readFile(char * filename)
{
    
    ifstream file(filename);
    
    //fp.seekg(0, std::ios::end);
    file.seekg(0, std::ios::end);

    size_t filesize = file.tellg();
    //fp.seekg(0, std::ios::beg);
    file.seekg(0, std::ios::beg);

    size_t numPackets = filesize/ 8256;

    //packet * packets = (packet *)malloc(sizeof(packet) * numPackets);
    std::vector<packet> packets(numPackets);

    for(int i = 0; i < numPackets; i++){
        // packets[i].adc1 = (int16_t *)malloc(sizeof(int16_t) * 1024);
        // packets[i].adc2 = (int16_t *)malloc(sizeof(int16_t) * 1024);
        // packets[i].adc3 = (int16_t *)malloc(sizeof(int16_t) * 1024);
        // packets[i].adc4 = (int16_t *)malloc(sizeof(int16_t) * 1024);
        packets[i].adc1.resize(1024);
        packets[i].adc2.resize(1024);
        packets[i].adc3.resize(1024);
        packets[i].adc4.resize(1024);

    }


    for(int i = 0; i < numPackets; i++){
        for (int j = 0; j < 1024; j++){
            file.read(reinterpret_cast<char*>(&(packets[i].adc1[j])), sizeof(int16_t));
            file.read(reinterpret_cast<char*>(&(packets[i].adc2[j])), sizeof(int16_t));
            file.read(reinterpret_cast<char*>(&(packets[i].adc3[j])), sizeof(int16_t));
            file.read(reinterpret_cast<char*>(&(packets[i].adc4[j])), sizeof(int16_t));

        }
    }

    return packets;
}


#endif
