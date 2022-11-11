from math import log10, log2
import numpy as np
from random import uniform
from IPython.display import Audio
from typing import Callable, Iterable


def freq_to_semis_from_c0(f: float) -> float:
    """convert frequency in Hz to semitones from c0"""
    f0 = Tone.from_name("C0").freq
    return 12 * log2(f / f0)


def semis_from_c0_to_freq(s: float) -> float:
    """convert semitones from c0 to frequency"""
    f0 = Tone.from_name("C0").freq
    return f0 * 2 ** (s / 12)


class Tone:
    """sine component of sound"""

    # notes names
    notes = ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#")

    def __init__(self, freq: float, p: float = 1.0, phase: float = 0.0):
        self.freq = freq
        self.p = p
        self.phase = phase

    def get_pressure_level(self) -> float:
        """pressure level in dB"""
        return 20 * log10(self.p / 0.00002)

    def get_overtones(
        self, decay: Callable[[float, int], float], n: int
    ) -> List["Tone"]:
        """
        Returns a list of tone that are overtones of this tone based on decay.\n
        
        `decay` is the overtone formula, takes in a float `p` and a int `i`, returns a float which is the pressure of the `i`-th overtone. `p` is the pressure of the base sine wave. The first overtone (`i` = 1) is the base tone.\n
        `n` is the number of overtones
        """
        return [
            Tone(self.freq * i, decay(self.p, i), self.phase) for i in range(1, n + 1)
        ]

    @classmethod
    def from_name(cls, name: str, p: float = 1.0) -> "Tone":
        """get tone from note name, doesn't work with flats"""
        if not name[:-1] in Tone.notes:
            raise ValueError(f"Unknown note: {name[:-1]}")

        # get the octave
        if len(name) == 3:
            octave = int(name[2])
        else:
            octave = int(name[1])

        # get the index of the note on the keyboard
        keyNumber = Tone.notes.index(name[0:-1])
        if keyNumber < 3:
            keyNumber = keyNumber + 12 + ((octave - 1) * 12) + 1
        else:
            keyNumber = keyNumber + ((octave - 1) * 12) + 1

        # calculate freqentcy and return
        freq = 440 * 2 ** ((keyNumber - 49) / 12)
        return Tone(freq, p)


class Osc:
    """An basic oscillator, which can produce a waveform, and support multiple voices"""

    # how detuned the most detuned voice is, in semitones
    DETUNE_RANGE = 2
    # how loud the other voices compared to the middle voice(s).
    # 1 means they're at the same level
    BLEND = 1

    def __init__(
        self,
        decay: Callable[[float, int], float],
        voices: int = 1,
        detune: float = 0,
        phase: float = np.pi,
        phase_random_range: float = 1.0,
        p: float = 1.0,
        pitch: float = 0.0,
        num_overs: int = 1,
    ):
        """
        Create a new oscillator\n
        `decay`: the decay formula.\n
        `voices`: the number of voices.\n
        `detune`: Between 0 and 1, how detuned the voice is. A value of 1 means the most detuned voice is detuned to Osc.DETUNE_RANGE.\n
        `phase`: the base phase of the voices.\n
        `phase_random_range`: phase of the voices are random every time a frequency is played, 0 means every voice have the same `phase`, and 1 means random phase.\n
        `p`: pressure, how loud the sound is.\n
        `pitch`: the difference in semitone of the sound played by the oscilaltor from the base freq that is requested.\n 
        `num_overs`: the number of overtones.\n
        """
        self.decay = decay
        self.num_overs = num_overs
        self.voices = voices
        self.detune = detune
        self.phase = phase
        self.phase_random_range = phase_random_range
        self.pitch = pitch
        self.p = p

    def to_tones(self, freq: float) -> List[Tone]:
        """Returns the list of tones when `freq` is played"""

        # get the list of freqencies of the voices
        if self.voices == 1:
            f = semis_from_c0_to_freq(freq_to_semis_from_c0(freq) + self.pitch)
            voice_freqs = np.array([f])
        else:
            d_semis = np.linspace(
                -self.DETUNE_RANGE * self.detune,
                self.DETUNE_RANGE * self.detune,
                self.voices,
            )
            base_semi = freq_to_semis_from_c0(freq)
            voice_freqs = np.vectorize(semis_from_c0_to_freq)(
                d_semis + base_semi + self.pitch
            )

        # get the overtones of the voices
        res = []
        for i in range(self.voices):
            # voices have random phase
            phase = uniform(
                self.phase * (1 - self.phase_random_range),
                self.phase * (1 - self.phase_random_range)
                + self.phase_random_range * np.pi * 2,
            )

            if (self.voices % 2 == 1 and i == self.voices // 2) or (
                self.voices % 2 == 0 and i in (self.voices // 2, self.voices // 2 + 1)
            ):
                # if it's the middle voice (if self.voices is even the there're 2 middle voices)
                # then get the overtones without blending
                res += Tone(voice_freqs[i], self.p, phase).get_overtones(
                    self.decay, self.num_overs
                )
            else:
                # get the blended tones
                res += Tone(voice_freqs[i], self.p * self.BLEND, phase).get_overtones(
                    self.decay, self.num_overs
                )

        return res

    @classmethod
    def sine_decay(cls, p: float, i: int) -> float:
        """Only base freq"""
        return p if i == 1 else 0.0

    @classmethod
    def sine_osc(
        cls,
        voices: int = 1,
        detune=0,
        phase: float = np.pi,
        phase_random_range=1,
        p: float = 1,
        pitch=0,
        overs=1,
    ) -> "Osc":
        return Osc(
            decay=Osc.sine_decay,
            voices=voices,
            detune=detune,
            phase=phase,
            phase_random_range=phase_random_range,
            p=p,
            pitch=pitch,
            num_overs=overs,
        )


class Synth:
    """A synthesizer that has multiple oscillators."""

    def __init__(self, oscs: List[Osc]):
        self.oscs = oscs

    def get_tones(self, notes: Iterable) -> List[Tone]:
        """Return the tones produced by synthesizer if notes are played.
         Notes can be a list of note names (str) or frequency (float)"""
        tones = []
        for note in notes:
            for osc in self.oscs:
                if type(note) == str:
                    freq = Tone.from_name(note).freq
                else:
                    freq = note
                tones += osc.to_tones(freq)
        return tones

    def play(self, notes: Iterable[str], rate=44100, secs=5) -> Audio:
        """
        Returns the audio of the notes playing.\n
        `rate`: sampling rate, default is 44100Hz\n
        `secs`: length of the audio clip in seconds, default is 5 seconds
        """
        tones = self.get_tones(notes)
        return self.play_tones(tones, rate, secs)

    def play_tones(self, tones: List[Tone], rate=44100, secs=5) -> Audio:
        """
        Returns the audio of the tones playing.\n
        `rate`: sampling rate, default is 44100Hz.\n
        `secs`: length of the audio clip in seconds, default is 5 seconds
        """
        ts = np.linspace(0, secs, rate * secs)
        data = np.sum(
            [
                tone.p * (2 ** 0.5) * np.sin(2 * np.pi * tone.freq * ts + tone.phase)
                for tone in tones
            ],
            axis=0,
        )
        return Audio(data, rate=rate)

