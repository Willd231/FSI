CXX = g++
CPPFLAGS=-g $(shell root-config --cflags)

CheckCount: CheckCount.cpp
    $(CC) $(CFLAGS) -o $@ $<

.PHONY: clean
clean:
    rm -f CheckCount
