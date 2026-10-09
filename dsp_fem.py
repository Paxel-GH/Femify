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


def build_sos(eq_gains_db, samplerate, deharsh_on=True):
    """Build unified SOS bank: [DC blocker, 5x EQ, optional de-harsh].

    Single pass = less CPU and no wandering between separate filter states.
    """
    sections = []
    # 1. DC blocker first — stops stacked lowshelves + 808s drifting
    sections.append(_biquad(10.0, samplerate, 0.0, "highpass", 0.707))
    for (_, f0, ftype, sq), g in zip(BANDS, eq_gains_db):
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
    """Gain + EQ + Femtanyl extras + transparent ceiling limiter."""

    # Transparent: only peaks OVER 1.0 get shaped, everything below passes
    # untouched. Memoryless tanh shoulder — no attack/release, no pumping.
    SOFT_KNEE = 1.00
    SOFT_CEIL = 0.99
    MONO_FREQ = 120.0
    DEHARSH_FREQ = 4000.0
    DEHARSH_DB = -3.0
    WIDEN_AMOUNT = 0.30

    def __init__(self, samplerate=48000, channels=2, gain=GAIN_DEFAULT,
                 eq_gains=None, limiter_enabled=True,
                 mono_bass=True, deharsh=True, widen=False):
        self.samplerate = samplerate
        self.channels = channels
        self.lock = Lock()
        self.gain = float(np.clip(gain, GAIN_MIN, GAIN_MAX))
        self.eq_gains = list(eq_gains) if eq_gains else [0.0] * 5
        self.limiter_enabled = bool(limiter_enabled)
        self.mono_bass = bool(mono_bass)
        self.deharsh = bool(deharsh)
        self.widen = bool(widen)
        self.sos = build_sos(self.eq_gains, samplerate, self.deharsh)
        b0, b1, b2, a1, a2 = _biquad(self.MONO_FREQ, samplerate, 0.0,
                                     "lowpass", 0.707)
        self.mono_sos = np.array([[b0, b1, b2, 1.0, a1, a2]], dtype=np.float64)
        self.zi = [np.zeros((self.sos.shape[0], 2), dtype=np.float64)
                   for _ in range(channels)]
        self.zi_mono = [np.zeros((1, 2), dtype=np.float64)
                        for _ in range(channels)]
        self._clip_flag = False

    def _rebuild_bank(self):
        new_sos = build_sos(self.eq_gains, self.samplerate, self.deharsh)
        if new_sos.shape[0] != self.sos.shape[0]:
            self.zi = [np.zeros((new_sos.shape[0], 2), dtype=np.float64)
                       for _ in range(self.channels)]
        self.sos = new_sos

    # ---- UI thread setters ----
    def set_gain(self, g):
        g = float(np.clip(g, GAIN_MIN, GAIN_MAX))
        with self.lock:
            self.gain = g

    def set_eq(self, gains):
        gains = [float(x) for x in gains]
        with self.lock:
            self.eq_gains = gains
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

    def get_state(self):
        with self.lock:
            return (self.gain, list(self.eq_gains), self.limiter_enabled,
                    self.mono_bass, self.deharsh, self.widen)

    def reset(self):
        with self.lock:
            self.zi = [np.zeros((self.sos.shape[0], 2), dtype=np.float64)
                       for _ in range(self.channels)]
            self.zi_mono = [np.zeros((1, 2), dtype=np.float64)
                            for _ in range(self.channels)]
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

        # Transparent ceiling limiter: peaks <= 1.0 pass bit-clean (other
        # than EQ), only overs get a tanh shoulder to 0.99. No time
        # constants, so loud sections keep RMS instead of ducking.
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
