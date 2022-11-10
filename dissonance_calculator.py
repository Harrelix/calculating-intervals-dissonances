import pandas as pd
from math import log10
import numpy as np
from itertools import combinations

params = pd.read_csv("ELLC-params.csv")

fs = params.loc[:, "f"].values
af = params.loc[:, "af"].values
Lu = params.loc[:, "Lu"].values
Tf = params.loc[:, "Tf"].values


def get_loudness(f, l_p):
    """Formulas and parameters taken from ISO 226:2003"""
    if f > max(fs) or f < min(fs):
        raise ValueError(
            f"Frequency {f:.2f} Not in bound. Has to be in [{min(fs):.2f}, {max(fs):.2f}]"
        )

    f1 = fs[0]
    f2 = fs[1]
    i2 = 1
    while f2 < f:
        f1 = f2
        i2 += 1
        f2 = fs[i2]
    i1 = i2 - 1

    a_f = af[i1] + (af[i2] - af[i1]) * (f - f1) / (f2 - f1)
    l_u = Lu[i1] + (Lu[i2] - Lu[i1]) * (f - f1) / (f2 - f1)
    t_f = Tf[i1] + (Tf[i2] - Tf[i1]) * (f - f1) / (f2 - f1)

    B_f = (
        (0.4 * 10 ** ((l_p + l_u) / 10 - 9)) ** a_f
        - (0.4 * 10 ** ((t_f + l_u) / 10 - 9)) ** a_f
        + 0.005135
    )
    return (40 * log10(B_f)) + 94


def critical_bandwidth(f):
    """Formula from Völk, 2015"""
    f /= 1000
    f_g_z = 25 + 75 * (1 + 1.4 * f ** 2) ** 0.69
    f_g_v = f_g_z * (1 - 1 / ((38.75 * f) ** 2 + 1))
    return f_g_v


def dissonance_amp(f1, f2):
    if f1 == f2:
        return 0
    f_bar = (f1 + f2) / 2
    x = abs(f1 - f2) / critical_bandwidth(f_bar)
    if x >= 1.2:
        return 0
    g = 4.906 * x * (1.2 - x) ** 4
    return g


def dissonance_total(freqs, ps):
    """Formulas from Dillion, 2013"""
    d_tot = 0
    for i, j in combinations(range(len(freqs)), 2):
        f1 = freqs[i]
        f2 = freqs[j]
        p1 = abs(ps[i])
        p2 = abs(ps[j])

        l_p1 = 20 * log10(p1 / 0.00002)
        l_p2 = 20 * log10(p2 / 0.00002)
        d_tot += (
            get_loudness(f1, l_p1) * get_loudness(f2, l_p2)
        ) ** 3 * dissonance_amp(f1, f2) ** 6
    d_tot **= 1 / 6
    return d_tot


def freq_from_note(note):
    notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

    if len(note) == 3:
        octave = int(note[2])
    else:
        octave = int(note[1])

    keyNumber = notes.index(note[0:-1])

    if keyNumber < 3:
        keyNumber = keyNumber + 12 + ((octave - 1) * 12) + 1
    else:
        keyNumber = keyNumber + ((octave - 1) * 12) + 1

    return 440 * 2 ** ((keyNumber - 49) / 12)

