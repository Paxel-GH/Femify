"""DSP for Femify: gain (1.5x-5.0x) -> 5-band EQ -> extras -> soft limiter.

Fresh rewrite inspired by Spotify Enhancer's dsp.py, retuned for Femtanyl:
distorted 808s, fast breaks, pitched-up vocals, already-hot masters.

Chain per block:
  1. gain (hearing range, default 2.0x)
  2. DC blocker (10 Hz highpass, always on — kills low-end wander)
  3. 5-band graphic EQ + optional de-harsh @4kHz in ONE filter pass
  4. mono bass <120Hz (optional, on by default)
  5. highs-only stereo widen (optional, off by default)
  6. musical ceiling limiter: gentle tanh shoulder from 0.80 to 0.99 —
     transients below the knee pass untouched (punch kept), only real
     overs get progressively rounded. Memoryless (no attack/release),
     so no pumping.

Real-time safe: coefficients recomputed only on setter calls,
filter state (zi) persists across blocks, audio thread never blocks.
Overflow guard resets filter state instead of blowing up.
"""
import numpy as np
from threading import Lock

try:
    from scipy.signal import sosfilt, sosfilt_zi, tf2sos
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    sosfilt = None  # type: ignore

# Femtanyl-shifted 5-band layout: (label, freq, type, Q/S)
# Lower sub (55 vs 60), tighter low-mid (180 vs 230), vocal at 900,
# harshness notch zone at 3500, air at 15000.
BANDS = [
    ("Sub", 55, "lowshelf", 1.0),
    ("Bass", 180, "peak", 1.1),
    ("Vox", 900, "peak", 1.0),
    ("Harsh", 3500, "peak", 1.2),
    ("Air", 15000, "highshelf", 1.0),
]

# Presets tuned around: ALL THE TIME / KATAMARI / MAN BITES DOG (album) /
# LOVESICK, CANNIBAL! — plus Flat. Order matches BANDS above.
FEM_PRESETS = {
    "Flat": [0.0, 0.0, 0.0, 0.0, 0.0],
    "ALL THE TIME": [5.0, 2.0, 1.0, 0.0, 3.0],
    "KATAMARI": [6.0, 3.0, -1.0, -1.0, 2.0],
    "MAN BITES DOG": [4.0, 0.0, 2.0, -2.5, 3.5],
    "LOVESICK, CANNIBAL!": [4.5, 1.0, 3.0, -1.0, 4.0],
    "PUSH UR T3MPRR": [6.0, 2.5, 0.5, -0.5, 3.0],
}

# Advanced 10-band layout: each normal band maps to its 2 nearest
# advanced bands (pairs below), values x3. Wider coverage + finer control.
ADV_BANDS = [
    ("Sub-", 31, "lowshelf", 1.0),
    ("Sub+", 62, "peak", 1.1),
    ("Low", 125, "peak", 1.1),
    ("LowMid", 250, "peak", 1.1),
    ("Mid-", 500, "peak", 1.0),
    ("Mid+", 1000, "peak", 1.0),
    ("HiMid-", 2000, "peak", 1.1),
    ("HiMid+", 4000, "peak", 1.2),
    ("Presence", 8000, "peak", 1.1),
    ("Air", 16000, "highshelf", 1.0),
]

# normal-band index -> (advanced-band pair)
ADV_MAP = {0: (0, 1), 1: (2, 3), 2: (4, 5), 3: (6, 7), 4: (8, 9)}

EQ_MAX_NORMAL = 12.0
EQ_MAX_ADV = 32.0


def _scale_presets_adv():
    out = {}
    for name, vals in FEM_PRESETS.items():
        adv = [0.0] * len(ADV_BANDS)
        for ni, v in enumerate(vals):
            for ai in ADV_MAP[ni]:
                adv[ai] = min(EQ_MAX_ADV, max(-EQ_MAX_ADV, float(v) * 3.0))
        out[name] = adv
    return out


ADV_PRESETS = _scale_presets_adv()

GAIN_MIN, GAIN_MAX = 1.50, 5.00
GAIN_DEFAULT = 2.00




def _biquad(f0, fs, db_gain, ftype, S_Q=1.0):
    """RBJ cookbook biquad. Returns (b0,b1,b2,a1,a2) with a0=1."""
    A = 10.0 ** (db_gain / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)

    if ftype == "peak":
        Q = max(0.1, S_Q)
        alpha = sin_w0 / (2.0 * Q)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A
    elif ftype in ("lowshelf", "highshelf"):
        S = max(0.1, S_Q)
        alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)
        sqrtA = np.sqrt(A)
        if ftype == "lowshelf":
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha)
            a0 = (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
            a2 = (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha
        else:
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha)
            a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
            a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha
    elif ftype == "lowpass":
        Q = max(0.1, S_Q)
        alpha = sin_w0 / (2.0 * Q)
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif ftype == "highpass":
        Q = max(0.1, S_Q)
        alpha = sin_w0 / (2.0 * Q)
        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    else:
        raise ValueError(ftype)

    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def build_sos(eq_gains_db, samplerate, deharsh_on=True, bands=None):
    """Build unified SOS bank: [DC blocker, Nx EQ, optional de-harsh].

    Single pass = less CPU and no wandering between separate filter states.
    bands defaults to the normal 5-band layout; advanced passes ADV_BANDS.
    """
    if bands is None:
        bands = BANDS
    sections = []
    # 1. DC blocker first — stops stacked lowshelves + 808s drifting
    sections.append(_biquad(10.0, samplerate, 0.0, "highpass", 0.707))
    for (_, f0, ftype, sq), g in zip(bands, eq_gains_db):
        sections.append(_biquad(f0, samplerate, float(g), ftype, sq))
    if deharsh_on:
        sections.append(_biquad(4000.0, samplerate, -3.0, "peak", 1.4))
    sos = np.zeros((len(sections), 6), dtype=np.float64)
    for i, (b0, b1, b2, a1, a2) in enumerate(sections):
        sos[i] = (b0, b1, b2, 1.0, a1, a2)
    return sos


def _single_sos(f0, fs, db_gain, ftype, sq=1.0):
    b0, b1, b2, a1, a2 = _biquad(f0, fs, db_gain, ftype, sq)
    return np.array([[b0, b1, b2, 1.0, a1, a2]], dtype=np.float64)


def find_device(sd_module, name, kind, hostapi_name="WASAPI"):
    """Exact match wins, else substring. Same behavior as original."""
    channel_key = f"max_{kind}_channels"
    wanted = name.casefold()
    hostapis = sd_module.query_hostapis()
    exact, partial, candidates = [], [], []
    for index, dev in enumerate(sd_module.query_devices()):
        if dev[channel_key] < 1:
            continue
        api_name = hostapis[dev["hostapi"]]["name"]
        if hostapi_name.casefold() not in api_name.casefold():
            continue
        candidates.append(f"  [{index}] {dev['name']}")
        dev_name = dev["name"].casefold()
        if dev_name == wanted:
            exact.append(index)
        elif wanted in dev_name:
            partial.append(index)
    matches = exact or partial
    if not matches:
        raise RuntimeError(
            f"No {hostapi_name} {kind} device found matching '{name}'.\n"
            f"Available {kind} devices:\n" + "\n".join(candidates)
        )
    return matches[0]


def list_wasapi_devices(sd_module, kind, hostapi_name="WASAPI"):
    """Return [(index, name)] for the device picker combos."""
    channel_key = f"max_{kind}_channels"
    hostapis = sd_module.query_hostapis()
    out = []
    for index, dev in enumerate(sd_module.query_devices()):
        if dev[channel_key] < 1:
            continue
        try:
            api_name = hostapis[dev["hostapi"]]["name"]
        except Exception:
            continue
        if hostapi_name.casefold() not in api_name.casefold():
            continue
        out.append((index, dev["name"]))
    return out


class AudioProcessor:
    """Gain + EQ + Femtanyl extras + musical ceiling limiter."""

    # Musical, not flat: knee well below 0dB with a WIDE tanh shoulder so
    # loud passages get progressively rounded (soft saturation harmonics)
    # instead of flat-topped. span MUST stay positive (knee < ceil) —
    # knee >= ceil degenerates into a brickwall hard clipper.
    SOFT_KNEE = 0.80
    SOFT_CEIL = 0.99
    MONO_FREQ = 120.0
    DEHARSH_FREQ = 4000.0
    DEHARSH_DB = -3.0
    WIDEN_AMOUNT = 0.30

    def __init__(self, samplerate=48000, channels=2, gain=GAIN_DEFAULT,
                 eq_gains=None, limiter_enabled=True,
                 mono_bass=True, deharsh=True, widen=False,
                 advanced=False, vsub_enabled=False,
                 vsub_level=0.6, vsub_harmonic=0.4,
                 vsub_crossover=120.0, vsub_drive=0.5,
                 vsub_resonance=0.5):
        self.samplerate = samplerate
        self.channels = channels
        self.lock = Lock()
        self.gain = float(np.clip(gain, GAIN_MIN, GAIN_MAX))
        self.advanced = bool(advanced)
        self.eq_gains = list(eq_gains) if eq_gains else [0.0] * self.n_bands
        self.limiter_enabled = bool(limiter_enabled)
        self.mono_bass = bool(mono_bass)
        self.deharsh = bool(deharsh)
        self.widen = bool(widen)
        # virtual subwoofer
        self.vsub_enabled = bool(vsub_enabled)
        self.vsub_level = float(np.clip(vsub_level, 0.0, 1.0))
        self.vsub_harmonic = float(np.clip(vsub_harmonic, 0.0, 1.0))
        self.vsub_crossover = float(np.clip(vsub_crossover, 60.0, 300.0))
        self.vsub_drive = float(np.clip(vsub_drive, 0.0, 1.0))
        self.vsub_resonance = float(np.clip(vsub_resonance, 0.0, 1.0))
        self.sos = build_sos(self.eq_gains, samplerate, self.deharsh,
                             self.active_bands)
        # vsub filter states
        self.vsub_lp_zi = [np.zeros((1, 2), dtype=np.float64)
                           for _ in range(channels)]
        self.vsub_sub_lp_zi = [np.zeros((1, 2), dtype=np.float64)
                               for _ in range(channels)]
        self.vsub_res_zi = [np.zeros((1, 2), dtype=np.float64)
                            for _ in range(channels)]
        self._rebuild_vsub_filters()
        b0, b1, b2, a1, a2 = _biquad(self.MONO_FREQ, samplerate, 0.0,
                                     "lowpass", 0.707)
        self.mono_sos = np.array([[b0, b1, b2, 1.0, a1, a2]], dtype=np.float64)
        self.zi = [np.zeros((self.sos.shape[0], 2), dtype=np.float64)
                   for _ in range(channels)]
        self.zi_mono = [np.zeros((1, 2), dtype=np.float64)
                        for _ in range(channels)]
        self._clip_flag = False

    @property
    def active_bands(self):
        return ADV_BANDS if self.advanced else BANDS

    @property
    def n_bands(self):
        return len(self.active_bands)

    @property
    def eq_max(self):
        return EQ_MAX_ADV if self.advanced else EQ_MAX_NORMAL

    def _rebuild_bank(self):
        new_sos = build_sos(self.eq_gains, self.samplerate, self.deharsh,
                            self.active_bands)
        if new_sos.shape[0] != self.sos.shape[0]:
            self.zi = [np.zeros((new_sos.shape[0], 2), dtype=np.float64)
                       for _ in range(self.channels)]
        self.sos = new_sos

    def _rebuild_vsub_filters(self):
        sr = self.samplerate
        co = self.vsub_crossover
        # low-pass at crossover for bass extraction
        b0, b1, b2, a1, a2 = _biquad(co, sr, 0.0, "lowpass", 0.707)
        self.vsub_lp_sos = np.array([[b0, b1, b2, 1.0, a1, a2]],
                                    dtype=np.float64)
        # low-pass at crossover/2 for subharmonic isolation
        sub_co = max(30.0, co / 2.0)
        b0, b1, b2, a1, a2 = _biquad(sub_co, sr, 0.0, "lowpass", 0.707)
        self.vsub_sub_lp_sos = np.array([[b0, b1, b2, 1.0, a1, a2]],
                                        dtype=np.float64)
        # peaking EQ at 65Hz for port resonance (gain scaled by resonance)
        res_gain = self.vsub_resonance * 12.0
        b0, b1, b2, a1, a2 = _biquad(65.0, sr, res_gain, "peak", 2.0)
        self.vsub_res_sos = np.array([[b0, b1, b2, 1.0, a1, a2]],
                                     dtype=np.float64)
        # reset filter states
        for ch in range(self.channels):
            self.vsub_lp_zi[ch][:] = 0.0
            self.vsub_sub_lp_zi[ch][:] = 0.0
            self.vsub_res_zi[ch][:] = 0.0

    # ---- UI thread setters ----
    def set_gain(self, g):
        g = float(np.clip(g, GAIN_MIN, GAIN_MAX))
        with self.lock:
            self.gain = g

    def set_eq(self, gains):
        lim = self.eq_max
        gains = [float(min(lim, max(-lim, x))) for x in gains]
        with self.lock:
            self.eq_gains = gains
            self._rebuild_bank()

    def set_eq_mode(self, advanced):
        """Switch normal 5-band <-> advanced 10-band. Caller must set_eq
        with the other mode's values right after (lengths differ)."""
        with self.lock:
            advanced = bool(advanced)
            if advanced != self.advanced:
                self.advanced = advanced
                n = len(ADV_BANDS if advanced else BANDS)
                self.eq_gains = (list(self.eq_gains)[:n] +
                                 [0.0] * max(0, n - len(self.eq_gains)))
                self._rebuild_bank()

    def set_limiter_enabled(self, on):
        with self.lock:
            self.limiter_enabled = bool(on)

    def set_mono_bass(self, on):
        with self.lock:
            self.mono_bass = bool(on)

    def set_deharsh(self, on):
        with self.lock:
            on = bool(on)
            if on != self.deharsh:
                self.deharsh = on
                self._rebuild_bank()

    def set_widen(self, on):
        with self.lock:
            self.widen = bool(on)

    def set_vsub_enabled(self, on):
        with self.lock:
            self.vsub_enabled = bool(on)

    def set_vsub_level(self, v):
        with self.lock:
            self.vsub_level = float(np.clip(v, 0.0, 1.0))

    def set_vsub_harmonic(self, v):
        with self.lock:
            self.vsub_harmonic = float(np.clip(v, 0.0, 1.0))

    def set_vsub_crossover(self, v):
        with self.lock:
            self.vsub_crossover = float(np.clip(v, 60.0, 300.0))
            self._rebuild_vsub_filters()

    def set_vsub_drive(self, v):
        with self.lock:
            self.vsub_drive = float(np.clip(v, 0.0, 1.0))

    def set_vsub_resonance(self, v):
        with self.lock:
            self.vsub_resonance = float(np.clip(v, 0.0, 1.0))
            self._rebuild_vsub_filters()




    def get_state(self):
        with self.lock:
            return (self.gain, list(self.eq_gains), self.limiter_enabled,
                    self.mono_bass, self.deharsh, self.widen, self.advanced,
                    self.vsub_enabled, self.vsub_level, self.vsub_harmonic,
                    self.vsub_crossover, self.vsub_drive, self.vsub_resonance)

    def reset(self):
        with self.lock:
            self.zi = [np.zeros((self.sos.shape[0], 2), dtype=np.float64)
                       for _ in range(self.channels)]
            self.zi_mono = [np.zeros((1, 2), dtype=np.float64)
                            for _ in range(self.channels)]
            for ch in range(self.channels):
                self.vsub_lp_zi[ch][:] = 0.0
                self.vsub_sub_lp_zi[ch][:] = 0.0
                self.vsub_res_zi[ch][:] = 0.0
            self._clip_flag = False

    def poll_clip(self):
        with self.lock:
            v = self._clip_flag
            self._clip_flag = False
            return v

    def _apply_sos_bank(self, audio, sos, zi_list):
        if _HAS_SCIPY:
            for ch in range(min(self.channels, audio.shape[1])):
                y, zf = sosfilt(sos, audio[:, ch], zi=zi_list[ch])
                audio[:, ch] = y
                zi_list[ch] = zf
        else:
            for ch in range(min(self.channels, audio.shape[1])):
                x = audio[:, ch]
                for (b0, b1, b2, _, a1, a2) in sos:
                    y = np.empty_like(x)
                    x1 = x2 = y1 = y2 = 0.0
                    for n in range(len(x)):
                        x0 = x[n]
                        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                        y[n] = y0
                        x2, x1, y2, y1 = x1, x0, y1, y0
                    x = y
                audio[:, ch] = x
        return audio

    # ---- audio thread ----
    def process(self, indata):
        with self.lock:
            gain = self.gain
            sos = self.sos.copy()
            limiter_on = self.limiter_enabled
            mono_on = self.mono_bass
            widen_on = self.widen

        audio = np.asarray(indata, dtype=np.float32) * gain

        # Single unified pass: DC blocker + EQ + de-harsh. Skip only when
        # everything is flat AND de-harsh off (DC blocker still runs).
        audio = audio.astype(np.float64, copy=True)
        audio = self._apply_sos_bank(audio, sos, self.zi)
        audio = audio.astype(np.float32, copy=False)

        # Overflow guard: hot IIR states can blow up at 5x + stacked bass.
        # Reset instead of letting levels wander (the "dipping" sound).
        if not np.all(np.isfinite(audio)):
            self.reset()
            return np.zeros_like(audio, dtype=np.float32)

        if (mono_on or widen_on) and audio.shape[1] >= 2:
            tmp = audio.astype(np.float64, copy=False)
            lows = np.empty_like(tmp)
            if _HAS_SCIPY:
                for ch in range(2):
                    y, zf = sosfilt(self.mono_sos, tmp[:, ch],
                                    zi=self.zi_mono[ch])
                    lows[:, ch] = y
                    self.zi_mono[ch] = zf
            else:
                lows[:] = tmp * 0.2
            if not np.all(np.isfinite(lows)):
                self.reset()
                return np.zeros_like(audio, dtype=np.float32)
            highs = tmp - lows
            if mono_on:
                mono_low = (lows[:, 0] + lows[:, 1]) * 0.5
                lows[:, 0] = mono_low
                lows[:, 1] = mono_low
            if widen_on:
                mid = (highs[:, 0] + highs[:, 1]) * 0.5
                side0 = highs[:, 0] - mid
                side1 = highs[:, 1] - mid
                k = 1.0 + self.WIDEN_AMOUNT
                highs[:, 0] = mid + side0 * k
                highs[:, 1] = mid + side1 * k
            audio = (lows + highs).astype(np.float32, copy=False)

        # Virtual subwoofer: generate sub-bass + harmonics + port resonance.
        # Additive — original audio preserved, vsub content layered on top.
        with self.lock:
            vs_on = self.vsub_enabled
            vs_level = self.vsub_level
            vs_harm = self.vsub_harmonic
            vs_drive = self.vsub_drive
            vs_res = self.vsub_resonance
            vs_lp_sos = self.vsub_lp_sos.copy()
            vs_sub_lp_sos = self.vsub_sub_lp_sos.copy()
            vs_res_sos = self.vsub_res_sos.copy()

        if vs_on:
            n = audio.shape[0]
            for ch in range(min(self.channels, audio.shape[1])):
                ch_audio = audio[:, ch].astype(np.float64)

                # extract bass
                bass, zf = sosfilt(vs_lp_sos, ch_audio,
                                   zi=self.vsub_lp_zi[ch])
                self.vsub_lp_zi[ch] = zf
                bass = np.asarray(bass, dtype=np.float64)

                # subharmonic: half-wave rectify bass, low-pass at crossover/2
                hw = np.where(bass > 0, bass, 0.0)
                sub, zf = sosfilt(vs_sub_lp_sos, hw,
                                  zi=self.vsub_sub_lp_zi[ch])
                self.vsub_sub_lp_zi[ch] = zf
                sub = np.asarray(sub, dtype=np.float64) * 2.0

                # harmonic: soft-clip bass, subtract original for harmonic content
                drive_factor = 1.0 + vs_drive * 4.0
                denom = np.tanh(drive_factor)
                if abs(denom) < 1e-9:
                    denom = 1e-9
                clipped = np.tanh(bass * drive_factor) / denom
                harmonic = (clipped - bass) * vs_harm

                # port resonance: peaking EQ on bass, subtract for delta
                res_bass, zf = sosfilt(vs_res_sos, bass,
                                       zi=self.vsub_res_zi[ch])
                self.vsub_res_zi[ch] = zf
                res_bass = np.asarray(res_bass, dtype=np.float64)
                resonance = (res_bass - bass) * vs_res

                # mix additive
                audio[:, ch] = (ch_audio +
                                vs_level * sub +
                                harmonic +
                                resonance).astype(np.float32)

        # Musical ceiling limiter: knee 0.80 with a wide 0.19 tanh shoulder.
        # Below-knee transients pass bit-clean (punch kept); overs get
        # progressively rounded instead of flat-topped. Memoryless, so
        # loud sections keep their RMS instead of ducking.
        pre_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if limiter_on:
            knee, ceil = self.SOFT_KNEE, self.SOFT_CEIL
            if pre_peak > knee:
                mag = np.abs(audio)
                over = mag > knee
                if np.any(over):
                    span = ceil - knee
                    shaped = knee + span * np.tanh((mag[over] - knee) / span)
                    audio[over] = np.sign(audio[over]) * shaped
                    audio = audio.astype(np.float32, copy=False)
                with self.lock:
                    self._clip_flag = True
        else:
            if pre_peak >= 0.999:
                with self.lock:
                    self._clip_flag = True

        np.clip(audio, -0.999, 0.999, out=audio)
        return audio


# Virtual subwoofer presets: (level, harmonic, crossover, drive, resonance)
VSUB_PRESETS = {
    "Low": (0.30, 0.20, 100.0, 0.20, 0.20),
    "Med": (0.50, 0.40, 120.0, 0.40, 0.40),
    "High": (0.80, 0.60, 140.0, 0.60, 0.60),
    "Femified": (0.90, 0.70, 130.0, 0.80, 0.80),
}
