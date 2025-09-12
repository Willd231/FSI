#!/usr/bin/python3

import os
import sys
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

print("T1: after sys/os import", flush=True)

# ---------- Config ----------
file_path = "ADCout_clean.dat"
adc_channels = ['ADCA', 'ADCB', 'ADCC', 'ADCD']
ninp = 4
nchan = 512
timestamp_size = 4                    # float32 seconds since epoch (little-endian)
freq_block_size = 4 * nchan           # nchan float32
autospec_block_size = 4 * nchan * ninp  # ninp * nchan float32
record_size = timestamp_size + freq_block_size + autospec_block_size

# ---------- File checks ----------
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found.", flush=True)
    sys.exit(1)

file_size = os.stat(file_path).st_size
nspec = file_size // record_size
if nspec == 0:
    print("Error: No spectra found in the file.", flush=True)
    sys.exit(1)

print(f"There were {nspec} Spectra recorded.", flush=True)

# ---------- Load data ----------
Autospec = np.zeros((nspec, ninp, nchan), dtype=np.float32)
timestamps = []
freq = None  # will capture from first spectrum (assumed constant across file)

with open(file_path, "rb") as fp:
    for i in range(nspec):
        # timestamp (float32 little-endian)
        raw_ts = fp.read(timestamp_size)
        if len(raw_ts) != timestamp_size:
            print(f"Incomplete timestamp at spectrum {i}. Stopping.", flush=True)
            break
        t = struct.unpack('<f', raw_ts)[0]
        timestamps.append(t)

        # frequency axis
        f = np.fromfile(fp, dtype=np.float32, count=nchan)
        if f.size != nchan:
            print(f"Incomplete freq block at spectrum {i}. Stopping.", flush=True)
            break
        if freq is None:
            freq = f.copy()

        # autospectra (ninp x nchan)
        a = np.fromfile(fp, dtype=np.float32, count=nchan * ninp)
        if a.size != nchan * ninp:
            print(f"Incomplete autospec block at spectrum {i}. Stopping.", flush=True)
            break
        Autospec[i] = a.reshape(ninp, nchan)

# sanity
if not timestamps:
    print("No timestamps parsed; file may be truncated or format mismatch.", flush=True)
    sys.exit(1)
if freq is None:
    print("No frequency axis parsed; file may be truncated.", flush=True)
    sys.exit(1)

# ---------- Prep for plotting ----------
# avoid log(0)
Autospec[Autospec <= 0] = 1e-10
readable_times = [datetime.fromtimestamp(float(t)) for t in timestamps]
time_labels = [dt.strftime("%H:%M:%S") for dt in readable_times]

# downsample time axis for heavy plots (optional)
MAX_SPEC = 2000
if Autospec.shape[0] > MAX_SPEC:
    step = max(1, Autospec.shape[0] // MAX_SPEC)
    Aplot = Autospec[::step]
    t_idx = np.arange(0, len(readable_times), step)
    times_plot = [time_labels[i] for i in t_idx]
else:
    Aplot = Autospec
    times_plot = time_labels

# ---------- Plot ----------
fig1, ax1 = plt.subplots(ninp, 1, figsize=(11, 18), constrained_layout=True)
fig1.suptitle(readable_times[0].strftime("%Y-%m-%d"), fontsize=16)

for ch in range(ninp):
    img = 10.0 * np.log10(Aplot[:, ch, :].T)  # shape: nchan x nspec_view
    cax = ax1[ch].imshow(img, cmap="copper_r", aspect="auto", origin="upper")
    ax1[ch].set_title(adc_channels[ch])
    ax1[ch].set_ylabel("Channel")
    # x ticks (time)
    n_ticks = min(6, len(times_plot))
    ticks = np.linspace(0, len(times_plot) - 1, num=n_ticks, dtype=int) if len(times_plot) else []
    ax1[ch].set_xticks(ticks)
    ax1[ch].set_xticklabels([times_plot[i] for i in ticks], rotation=30, ha="right")
    ax1[ch].set_xlabel("Time")
    cb = fig1.colorbar(cax, ax=ax1[ch])
    cb.ax.set_ylabel("dB")

# ---------- Interactivity ----------
figure_closed = False

def on_close(event):
    nonlocal_flag = 'figure_closed' in globals()  # just to satisfy linters
    # update the global flag
    globals()['figure_closed'] = True
    print(f"Figure {event.canvas.figure.number} closed!", flush=True)

def plot_selected_spectrum(s_idx: int):
    if not (0 <= s_idx < nspec):
        print(f"Invalid spectrum number: {s_idx}", flush=True)
        return
    fig2, ax2 = plt.subplots(ninp, 1, figsize=(11, 18), constrained_layout=True)
    for i in range(ninp):
        ax2[i].plot(freq, 10.0 * np.log10(Autospec[s_idx, i, :]))
        ax2[i].set_title(f"{adc_channels[i]} - Spectrum {s_idx}")
        ax2[i].set_xlabel("Frequency (MHz)")
        ax2[i].set_ylabel("Amplitude (dB)")
    plt.show(block=False)

def on_click(event):
    # only respond to clicks inside the main spectrogram axes
    if event.inaxes in ax1 and event.xdata is not None:
        s_idx = int(round(event.xdata))
        if 0 <= s_idx < nspec:
            print(f"Selected spectrum: {s_idx}", flush=True)
            plot_selected_spectrum(s_idx)

fig1.canvas.mpl_connect('close_event', on_close)
fig1.canvas.mpl_connect('button_press_event', on_click)

backend = matplotlib.get_backend()
print("Backend:", backend, flush=True)

# Treat only true non-GUI backends as headless
NON_GUI = {"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"}
if backend.lower() in NON_GUI:
    print("Headless backend detected — saving to spectrogram.png", flush=True)
    fig1.savefig("spectrogram.png", dpi=150)
    sys.exit(0)

# GUI path: block so the window stays up, clicks work
plt.show()
