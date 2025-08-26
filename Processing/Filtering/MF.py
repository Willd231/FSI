#!/usr/bin/python3

def getData(filename):
    data = []
    with open(filename, "rb+") as fp:
        while fp != null:
            data.append(fp.read())
    return data


def median(A, B, C):
    sortedlist = sorted([A, B, C])
    return sortedlist[1]


def doFilter(Data):
    Data.insert(0,0)
    Filtered_Data = []
    for i in range(0, len(Data)):
        median = median(Data[i],Data[i+1], Data[i+2])
        Filtered_Data.append(median)

def writeFiltered(Filtered_Data):
    with open(Filtered_Data.dat, "wb+") as f:
        f.write(Data)



def Main(argc, argv):
    data = getData(argv[1])
    filtered_data = doFilter(data)
    writeFiltered(filtered_data)   




if " __name__" == "Main":
    Main()









