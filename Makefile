#  Makefile for PAF programs 
#
#  Anish Nov 06, 2010

CC = gcc
FF = g77 
FFLAGS = -lm 
CFLAGS = -lm 
CORFLG = -Wall -O3 -D_FILE_OFFSET_BITS=64 -lm  -I/opt/local/stow/fftw-3.2.2/include/
LPG =  -L/usr/lib64 -lpng12 -L/usr/lib64 -lpgplot  -L/usr/lib64 -lX11
FFTW =  -L/opt/local/lib/ -lfftw3f
AGGFLG = -D_FILE_OFFSET_BITS=64

corr: 
	@echo "   "
	@echo "Linking corr_cpu_complex ....."
	$(CC) $(CORFLG) -o corr corr_cpu_complex.c $(FFTW) $(AGGFLAG)  

agg:
	@echo "   "
	@echo "Linking aggregate ....   "
	$(CC) $(AGGFLG) -o aggregate aggregate.c   