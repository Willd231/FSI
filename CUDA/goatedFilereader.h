
#ifndef goatedFilereader_H
#define goatedFilereader_H
 
typedef struct packets{
int16_t * adc1, *adc2, *adc3, *adc4;
}
packets;




inline readFile(char * filename)

{
FILE * fp = fopen(fp, "rb");
packets * new = malloc(sizeof(packets));
 
int16 ** return = malloc(sizeof(
int filesize = 0;
fp.seekg(0, std::ios::end);
size_t filesize = fp.tellg();
fp.seekg(0, std::ios::beg);
size_t numPackets = filesize/ 8256;

new->adc1 = malloc(sizeof(int16_t) * 1024 * numPackets);
new->adc2 = malloc(sizeof(int16_t)* 1024 * numPackets);
new->adc3 = malloc(sizeof(int16_t)* 1024 * numPackets);
new->adc4 = malloc(sizeof(int16_t)* 1024 * numPackets); 

for(int i = 0; i < numPackets; i++){
for (int j = 0; j < 1024; j++){
fp.read(reinterpret_cast<char*>(&(new->adc1[j])), sizeof(int16_t));
fp.read(reinterpret_cast<char*>(&(new->adc2[j])), sizeof(int16_t));
fp.read(reinterpret_cast<char*>(&(new->adc3[j])), sizeof(int16_t));
fp.read(reinterpret_cast<char*>(&(new->adc4[j])), sizeof(int16_t));

}
}



#endif
