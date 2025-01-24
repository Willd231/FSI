#!/usr/bin/python3

import numpy as np


def write(filename="testout.dat"):
     
    values = np.random.rand(1024).astype(np.float64)
    with open(filename, "wb") as out:
        out.write(values.tobytes())  
    return values


def read(filename="testout.dat"):
    with open(filename, "rb") as inp:
        data = inp.read(1024 * 8)  
    return np.frombuffer(data, dtype=np.float64)  


def difference():
    written_vals = write()
    read_vals = read()

    if not np.allclose(written_vals, read_vals):
        print("Mismatch detected!")
        for i, (w, r) in enumerate(zip(written_vals, read_vals)):
            if not np.isclose(w, r):
                print(f"Mismatch at index {i}: Written val = {w}, Read val = {r}")
    else:
        print("everything is correct")


def main():
    difference()

main()
