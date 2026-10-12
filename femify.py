import json
import os
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QDoubleSpinBox, QGroupBox, QComboBox,
    QPushButton, QCheckBox, QSystemTrayIcon, QMenu, QMessageBox,
    QStatusBar, QButtonGroup, QStackedWidget, QDialog,
)

from dsp_fem import (
    AudioProcessor, BANDS, FEM_PRESETS, ADV_BANDS, ADV_PRESETS,
    EQ_MAX_NORMAL, EQ_MAX_ADV,
    find_device, list_wasapi_devices,
    GAIN_MIN, GAIN_MAX, GAIN_DEFAULT,
)
from theme import apply_theme, make_icon, ACCENT

SAMPLERATE = 48000
CHANNELS = 2
BLOCKSIZE = 1024
HOSTAPI_NAME = "WASAPI"

DEFAULT_INPUT = "CABLE Output (VB-Audio Virtual Cable)"
DEFAULT_OUTPUT = ""  # auto: first WASAPI output

CONFIG_PATH = Path(os.environ.get("APPDATA", str(Path.home()))) / "Femify" / "settings.json"
LEGACY_PATH = Path(os.environ.get("APPDATA", str(Path.home()))) / "SpotifyEnhancer" / "settings.json"

HEARING_WARN_AT = 3.5


def load_config():
    cfg = {
        "gain": GAIN_DEFAULT,
        "eq": [0.0] * 5,
        "adv_eq": [0.0] * 10,
        "eq_mode": "normal",
        "preset": "Flat",
        "limiter": True,
        "mono_bass": True,
        "deharsh": True,
        "widen": False,
        "input_name": DEFAULT_INPUT,
        "output_name": DEFAULT_OUTPUT,
    }
    # one-time migrate from Spotify Enhancer
    try:
        if not CONFIG_PATH.exists() and LEGACY_PATH.exists():
            old = json.loads(LEGACY_PATH.read_text())
            for k in ("gain", "eq", "preset", "limiter", "input_name", "output_name"):
                if k in old:
                    cfg[k] = old[k]
    except Exception:
        pass
    try:
        if CONFIG_PATH.exists():
            cfg.update(json.loads(CONFIG_PATH.read_text()))
    except Exception:
        pass
    try:
        cfg["gain"] = min(GAIN_MAX, max(GAIN_MIN, float(cfg.get("gain", GAIN_DEFAULT))))
    except Exception:
        cfg["gain"] = GAIN_DEFAULT
    eq = list(cfg.get("eq", [0.0] * 5))[:5]
    while len(eq) < 5:
        eq.append(0.0)
    cfg["eq"] = [min(EQ_MAX_NORMAL, max(-EQ_MAX_NORMAL, float(x))) for x in eq]
    adv = list(cfg.get("adv_eq", [0.0] * 10))[:10]
    while len(adv) < 10:
        adv.append(0.0)
    cfg["adv_eq"] = [min(EQ_MAX_ADV, max(-EQ_MAX_ADV, float(x))) for x in adv]
    if cfg.get("eq_mode") not in ("normal", "advanced"):
        cfg["eq_mode"] = "normal"
    return cfg


class AdvWarningDialog(QDialog):
    """Hearing-damage gate for advanced EQ. 5s countdown, Enter locked."""

    def __init__(self, parent, gain, countdown_seconds=5):
        super().__init__(parent)
        self.setWindowTitle("ADVANCED EQ — WARNING")
        self.setModal(True)
        self._left = countdown_seconds
        self.accepted_by_user = False

        root = QVBoxLayout(self)
        warn = QLabel(
            "!! ADVANCED EQ CAN CAUSE ACTUAL HARM TO YOUR EARS !!\n\n"
            f"10 bands up to ±32 dB, presets ×3"
            f" — couple that with your current {gain:.1f}x gain"
            + (" (2.5x+ DANGER ZONE)" if gain >= 2.5 else "")
            + " and peaks sit permanently at the ceiling.\n\n"
            "Start low, limiter recommended.\n"
            "You enter at your own risk."
        )
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #ff2a5f; font-weight: 900;")
        root.addWidget(warn)

        self.count_label = QLabel()
        self.count_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.count_label)

        row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.enter_btn = QPushButton()
        self.enter_btn.setEnabled(False)
        self.enter_btn.clicked.connect(self._enter)
        row.addWidget(cancel)
        row.addWidget(self.enter_btn)
        root.addLayout(row)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._tick()
        self._timer.start()

    def _tick(self):
        if self._left > 0:
            self.count_label.setText(f"Think about your ears… {self._left}s")
            self.enter_btn.setText(f"Enter ({self._left})")
            self._left -= 1
        else:
            self._timer.stop()
            self.count_label.setText("You may enter.")
            self.enter_btn.setText("I understand — Enter")
            self.enter_btn.setEnabled(True)

    def _enter(self):
        self.accepted_by_user = True
        self.accept()


def save_config(cfg):
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


class AudioEngine:
    """Single input -> single output. No BT fan-out (unlike original)."""

    def __init__(self, processor):
        self.proc = processor
        self.stream = None
        self.running = False

    def _audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(status, flush=True)
        outdata[:] = self.proc.process(indata)

    def start(self, in_idx, out_idx):
        self.stop()
        self.proc.reset()
        self.stream = sd.Stream(
            device=(in_idx, out_idx), samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE, channels=CHANNELS,
            dtype="float32", callback=self._audio_callback,
        )
        self.stream.start()
        self.running = True

    def stop(self):
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.stream = None
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.cfg = load_config()
        self.adv_mode = self.cfg.get("eq_mode", "normal") == "advanced"
        init_eq = self.cfg.get("adv_eq" if self.adv_mode else "eq")
        self.proc = AudioProcessor(
            samplerate=SAMPLERATE, channels=CHANNELS,
            gain=self.cfg["gain"], eq_gains=init_eq,
            limiter_enabled=self.cfg.get("limiter", True),
            mono_bass=self.cfg.get("mono_bass", True),
            deharsh=self.cfg.get("deharsh", True),
            widen=self.cfg.get("widen", False),
            advanced=self.adv_mode,
        )
        self.engine = AudioEngine(self.proc)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._save_now)
        self._updating = False

        self.setWindowTitle("Femify")
        self.setWindowIcon(make_icon())
        self.resize(620, 640)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- header ----
        title = QLabel("FEMIFY")
        title.setObjectName("femTitle")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)
        sub = QLabel("FEMTANYL MODE // 1.5x-5.0x MAXIMUM HEARING")
        sub.setObjectName("femSub")
        sub.setAlignment(Qt.AlignCenter)
        root.addWidget(sub)

        # ---- Gain ----
        gain_box = QGroupBox("GAIN (maximum hearing)")
        gl = QHBoxLayout(gain_box)
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(int(GAIN_MIN * 100), int(GAIN_MAX * 100))
        self.gain_slider.setValue(int(self.proc.gain * 100))
        self.gain_slider.setTickPosition(QSlider.TicksBelow)
        self.gain_slider.setTickInterval(50)
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(GAIN_MIN, GAIN_MAX)
        self.gain_spin.setSingleStep(0.10)
        self.gain_spin.setValue(self.proc.gain)
        self.gain_spin.setSuffix("x")
        self.gain_slider.valueChanged.connect(self._on_gain_slider)
        self.gain_spin.valueChanged.connect(self._on_gain_spin)
        gl.addWidget(QLabel("Volume"))
        gl.addWidget(self.gain_slider, 1)
        gl.addWidget(self.gain_spin)
        root.addWidget(gain_box)
        self.hearing_label = QLabel("")
        self.hearing_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.hearing_label)
        self._update_hearing_warning(self.proc.gain)

        # ---- EQ (normal 5-band / advanced 10-band pages) ----
        self.eq_box = QGroupBox("EQUALIZER (±12 dB)")
        eq_root = QVBoxLayout(self.eq_box)
        self.eq_stack = QStackedWidget()
        eq_root.addWidget(self.eq_stack)
        self.eq_pages = {}  # adv_flag -> {"sliders": [], "labels": []}
        self.eq_stack.addWidget(self._build_eq_page(BANDS, EQ_MAX_NORMAL, False))
        self.eq_stack.addWidget(self._build_eq_page(ADV_BANDS, EQ_MAX_ADV, True))
        # restore saved values into both pages
        self._set_page_values(False, self.cfg.get("eq", [0.0] * 5))
        self._set_page_values(True, self.cfg.get("adv_eq", [0.0] * 10))
        self.eq_stack.setCurrentIndex(1 if self.adv_mode else 0)
        self._refresh_eq_title()
        root.addWidget(self.eq_box)

        # ---- Preset buttons ----
        preset_box = QGroupBox("FEM PRESETS")
        pv = QVBoxLayout(preset_box)
        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(True)
        self.preset_buttons = {}
        row = QHBoxLayout()
        for j, name in enumerate(FEM_PRESETS.keys()):
            b = QPushButton(name)
            b.setObjectName("presetBtn")
            b.setCheckable(True)
            b.clicked.connect(lambda _c, n=name: self._apply_preset(n))
            row.addWidget(b)
            self.preset_group.addButton(b)
            self.preset_buttons[name] = b
            if j == 2:  # wrap to 2 rows of 3
                pv.addLayout(row)
                row = QHBoxLayout()
        pv.addLayout(row)
        adv_row = QHBoxLayout()
        adv_row.addWidget(QLabel("EQ mode:"))
        self.adv_btn = QPushButton("NORMAL EQ")
        self.adv_btn.setObjectName("presetBtn")
        self.adv_btn.setCheckable(True)
        self.adv_btn.setChecked(self.adv_mode)
        self.adv_btn.toggled.connect(self._on_adv_toggle)
        adv_row.addWidget(self.adv_btn, 1)
        self.adv_banner = QLabel("ADVANCED ±32dB ACTIVE — WATCH YOUR EARS")
        self.adv_banner.setAlignment(Qt.AlignCenter)
        self.adv_banner.setStyleSheet("color: #ff2222; font-weight: 900;")
        self.adv_banner.setVisible(self.adv_mode)
        adv_row.addWidget(self.adv_banner, 2)
        pv.addLayout(adv_row)
        if self.adv_mode:
            self.adv_btn.setText("ADVANCED EQ")
        custom_row = QHBoxLayout()
        custom_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(FEM_PRESETS.keys()) + ["Custom"])
        self.preset_combo.setCurrentText(self.cfg.get("preset", "Flat"))
        self.preset_combo.currentTextChanged.connect(self._on_preset)
        custom_row.addWidget(self.preset_combo, 1)
        self.limiter_check = QCheckBox("Soft limiter")
        self.limiter_check.setChecked(self.proc.limiter_enabled)
        self.limiter_check.toggled.connect(self._on_limiter)
        custom_row.addWidget(self.limiter_check)
        self.clip_label = QLabel("● OK")
        self.clip_label.setStyleSheet("color: #39ff6a; font-weight: bold;")
        custom_row.addWidget(self.clip_label)
        pv.addLayout(custom_row)
        root.addWidget(preset_box)
        self._resync_preset_ui()

        # ---- Extras ----
        extra_box = QGroupBox("FEM FLAVOR")
        el = QHBoxLayout(extra_box)
        self.mono_check = QCheckBox("Mono bass <120Hz")
        self.mono_check.setChecked(self.proc.mono_bass)
        self.mono_check.toggled.connect(lambda on: (self.proc.set_mono_bass(on), self.schedule_save()))
        self.deharsh_check = QCheckBox("De-harsh 4k")
        self.deharsh_check.setChecked(self.proc.deharsh)
        self.deharsh_check.toggled.connect(lambda on: (self.proc.set_deharsh(on), self.schedule_save()))
        self.widen_check = QCheckBox("Widen highs +30%")
        self.widen_check.setChecked(self.proc.widen)
        self.widen_check.toggled.connect(lambda on: (self.proc.set_widen(on), self.schedule_save()))
        el.addWidget(self.mono_check)
        el.addWidget(self.deharsh_check)
        el.addWidget(self.widen_check)
        root.addWidget(extra_box)

        # ---- Devices (single output) ----
        dev_box = QGroupBox("DEVICES (WASAPI)")
        dl = QVBoxLayout(dev_box)
        self.in_combo = QComboBox()
        self.out_combo = QComboBox()
        for text, combo in (("Capture from:", self.in_combo), ("Play to:", self.out_combo)):
            r = QHBoxLayout()
            r.addWidget(QLabel(text), 0)
            r.addWidget(combo, 1)
            dl.addLayout(r)
        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh list")
        refresh_btn.clicked.connect(self._refresh_devices)
        self.restart_btn = QPushButton("Restart audio")
        self.restart_btn.clicked.connect(self._restart_audio)
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(self.restart_btn)
        dl.addLayout(btn_row)
        root.addWidget(dev_box)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # ---- Tray ----
        self.tray = QSystemTrayIcon(make_icon(), self)
        self.tray.setToolTip("Femify")
        menu = QMenu()
        show_act = QAction("Show", self)
        show_act.triggered.connect(self.showNormal_raise)
        exit_act = QAction("Exit", self)
        exit_act.triggered.connect(self._exit)
        menu.addAction(show_act)
        menu.addAction(exit_act)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray)
        self.tray.show()

        self._clip_timer = QTimer(self)
        self._clip_timer.setInterval(150)
        self._clip_timer.timeout.connect(self._poll_clip)
        self._clip_timer.start()

        self._refresh_devices(select_saved=True)
        self._restart_audio(silent=False)

    # ---------- EQ pages ----------
    def _active_tables(self):
        if self.adv_mode:
            return ADV_BANDS, ADV_PRESETS, EQ_MAX_ADV
        return BANDS, FEM_PRESETS, EQ_MAX_NORMAL

    def _build_eq_page(self, bands, eq_max, adv_flag):
        page = QWidget()
        grid = QGridLayout(page)
        grid.setSpacing(4)
        sliders, labels = [], []
        rng = int(eq_max * 10)
        per_row = 5
        for i, (label, freq, _, _) in enumerate(bands):
            col = QVBoxLayout()
            slider = QSlider(Qt.Vertical)
            slider.setRange(-rng, rng)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksRight)
            slider.setTickInterval(rng // 2)
            slider.valueChanged.connect(lambda v, idx=i: self._on_eq(idx, v / 10.0))
            val = QLabel("+0.0 dB")
            val.setAlignment(Qt.AlignCenter)
            col.addWidget(slider, 1, alignment=Qt.AlignHCenter)
            col.addWidget(val)
            freq_txt = f"{freq}" if freq < 1000 else f"{freq/1000:g}k"
            name = QLabel(f"{label}\n{freq_txt}")
            name.setAlignment(Qt.AlignCenter)
            col.addWidget(name)
            grid.addLayout(col, i // per_row, i % per_row)
            sliders.append(slider)
            labels.append(val)
        self.eq_pages[adv_flag] = {"sliders": sliders, "labels": labels}
        return page

    def _set_page_values(self, adv_flag, values):
        info = self.eq_pages[adv_flag]
        for s, lab, v in zip(info["sliders"], info["labels"], values):
            s.blockSignals(True)
            s.setValue(int(v * 10))
            s.blockSignals(False)
            lab.setText(f"{v:+.1f} dB")

    def _page_values(self, adv_flag):
        return [s.value() / 10.0 for s in self.eq_pages[adv_flag]["sliders"]]

    def _refresh_eq_title(self):
        if self.adv_mode:
            self.eq_box.setTitle("ADVANCED EQUALIZER (±32 dB — DANGER)")
        else:
            self.eq_box.setTitle("EQUALIZER (±12 dB)")

    # ---------- UI slots ----------
    @Slot(int)
    def _on_gain_slider(self, v):
        if self._updating:
            return
        self._updating = True
        self.gain_spin.setValue(v / 100.0)
        self._updating = False
        self.proc.set_gain(v / 100.0)
        self._update_hearing_warning(v / 100.0)
        self.schedule_save()

    @Slot(float)
    def _on_gain_spin(self, v):
        if self._updating:
            return
        self._updating = True
        self.gain_slider.setValue(int(round(v * 100)))
        self._updating = False
        self.proc.set_gain(v)
        self._update_hearing_warning(v)
        self.schedule_save()

    def _update_hearing_warning(self, g):
        if self.adv_mode:
            self.hearing_label.setText(f"!! ADVANCED EQ ±32dB @ {g:.1f}x — EAR DAMAGE CERTAIN IF LOUD !!")
            self.hearing_label.setStyleSheet(f"color: {ACCENT}; font-weight: 900;")
        elif g >= HEARING_WARN_AT:
            self.hearing_label.setText(f"!! {g:.1f}x HEARING DAMAGE RANGE !!")
            self.hearing_label.setStyleSheet(f"color: {ACCENT}; font-weight: 900;")
        elif g >= 2.5:
            self.hearing_label.setText(f"{g:.1f}x loud — limiter recommended")
            self.hearing_label.setStyleSheet("color: #ffcc00; font-weight: bold;")
        else:
            self.hearing_label.setText("")
            self.hearing_label.setStyleSheet("")

    def _on_eq(self, idx, db):
        info = self.eq_pages[self.adv_mode]
        info["labels"][idx].setText(f"{db:+.1f} dB")
        eq = self._page_values(self.adv_mode)
        self.proc.set_eq(eq)
        _, presets, _ = self._active_tables()
        for name, vals in presets.items():
            if all(abs(a - b) < 0.05 for a, b in zip(eq, vals)):
                self.preset_combo.blockSignals(True)
                self.preset_combo.setCurrentText(name)
                self.preset_combo.blockSignals(False)
                self._sync_preset_buttons(name)
                break
        else:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Custom")
            self.preset_combo.blockSignals(False)
            self._sync_preset_buttons("Custom")
        self.schedule_save()

    def _sync_preset_buttons(self, name):
        for n, b in self.preset_buttons.items():
            b.blockSignals(True)
            b.setChecked(n == name)
            b.blockSignals(False)

    def _apply_preset(self, name):
        _, presets, _ = self._active_tables()
        vals = presets[name]
        self.proc.set_eq(vals)
        self._set_page_values(self.adv_mode, vals)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText(name)
        self.preset_combo.blockSignals(False)
        self._sync_preset_buttons(name)
        self.schedule_save()

    def _on_preset(self, name):
        _, presets, _ = self._active_tables()
        if name in presets:
            self._apply_preset(name)
        self.schedule_save()

    def _on_adv_toggle(self, checked):
        if checked == self.adv_mode:
            return
        if checked:
            # gate every entry behind the 5s warning
            gain = self.proc.gain
            dlg = AdvWarningDialog(self, gain)
            if dlg.exec() != QDialog.Accepted or not dlg.accepted_by_user:
                self.adv_btn.blockSignals(True)
                self.adv_btn.setChecked(False)
                self.adv_btn.blockSignals(False)
                return
            self._enter_advanced()
        else:
            self._exit_advanced()

    def _enter_advanced(self):
        # stash normal values, switch DSP + page, load advanced values
        self.cfg["eq"] = self._page_values(False)
        self.adv_mode = True
        self.proc.set_eq_mode(True)
        self.proc.set_eq(self._page_values(True))
        self.eq_stack.setCurrentIndex(1)
        self.adv_btn.setText("ADVANCED EQ")
        self.adv_banner.setVisible(True)
        self._refresh_eq_title()
        self._resync_preset_ui()
        self._update_hearing_warning(self.proc.gain)
        self.schedule_save()

    def _exit_advanced(self):
        self.cfg["adv_eq"] = self._page_values(True)
        self.adv_mode = False
        self.proc.set_eq_mode(False)
        self.proc.set_eq(self._page_values(False))
        self.eq_stack.setCurrentIndex(0)
        self.adv_btn.setText("NORMAL EQ")
        self.adv_banner.setVisible(False)
        self._refresh_eq_title()
        self._resync_preset_ui()
        self._update_hearing_warning(self.proc.gain)
        self.schedule_save()

    def _resync_preset_ui(self):
        _, presets, _ = self._active_tables()
        eq = self._page_values(self.adv_mode)
        for name, vals in presets.items():
            if all(abs(a - b) < 0.05 for a, b in zip(eq, vals)):
                self.preset_combo.blockSignals(True)
                self.preset_combo.setCurrentText(name)
                self.preset_combo.blockSignals(False)
                self._sync_preset_buttons(name)
                return
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Custom")
        self.preset_combo.blockSignals(False)
        self._sync_preset_buttons("Custom")

    def _on_limiter(self, on):
        # never forced — user choice even at 5x
        self.proc.set_limiter_enabled(on)
        self.schedule_save()

    # ---------- devices / engine ----------
    def _refresh_devices(self, select_saved=False):
        try:
            ins = list_wasapi_devices(sd, "input", HOSTAPI_NAME)
            outs = list_wasapi_devices(sd, "output", HOSTAPI_NAME)
        except Exception as e:
            self.status.showMessage(f"Device query failed: {e}")
            return
        for combo, devs in ((self.in_combo, ins), (self.out_combo, outs)):
            combo.blockSignals(True)
            combo.clear()
            for idx, name in devs:
                combo.addItem(f"[{idx}] {name}", userData=name)
            combo.blockSignals(False)
        if select_saved:
            self._select_by_name(self.in_combo, self.cfg.get("input_name", ""))
            saved_out = self.cfg.get("output_name", "")
            if saved_out:
                self._select_by_name(self.out_combo, saved_out)
            # else: leave at first output (auto)

    def _select_by_name(self, combo, name):
        if not name:
            return
        for i in range(combo.count()):
            if combo.itemData(i) == name or name.casefold() in combo.itemText(i).casefold():
                combo.setCurrentIndex(i)
                return

    def _restart_audio(self, silent=True):
        in_name = self.in_combo.currentData() or self.cfg.get("input_name", DEFAULT_INPUT)
        out_name = self.out_combo.currentData() or self.cfg.get("output_name", "")
        try:
            in_idx = find_device(sd, in_name, "input", HOSTAPI_NAME)
            if not out_name:
                # auto-pick current combo selection
                out_name = self.out_combo.currentData() or ""
                if not out_name:
                    raise RuntimeError("No output device available.")
            out_idx = find_device(sd, out_name, "output", HOSTAPI_NAME)
            self.engine.start(in_idx, out_idx)
            self.status.showMessage(f"Running: {in_name} → {out_name} @ {self.proc.gain:.1f}x")
            self.cfg.update({"input_name": in_name, "output_name": out_name})
            self.schedule_save()
        except Exception as e:
            self.engine.stop()
            self.status.showMessage("Audio stopped — check devices")
            if not silent:
                QMessageBox.warning(self, "Femify audio error",
                                    f"Could not start audio.\n\n{e}\n\nPick devices and press Restart audio.")

    # ---------- persistence ----------
    def schedule_save(self):
        self._save_timer.start()

    def _save_now(self):
        gain, eq, lim, mono, deh, wid, adv = self.proc.get_state()
        # stash the visible page's live values into the right key
        if adv:
            self.cfg["adv_eq"] = self._page_values(True)
        else:
            self.cfg["eq"] = self._page_values(False)
        self.cfg.update({
            "gain": gain, "limiter": lim,
            "mono_bass": mono, "deharsh": deh, "widen": wid,
            "eq_mode": "advanced" if adv else "normal",
            "preset": self.preset_combo.currentText(),
            "input_name": self.in_combo.currentData() or self.cfg.get("input_name", ""),
            "output_name": self.out_combo.currentData() or self.cfg.get("output_name", ""),
        })
        save_config(self.cfg)

    # ---------- tray / window ----------
    def _on_tray(self, reason):
        if reason in (QSystemTrayIcon.DoubleClick, QSystemTrayIcon.Trigger):
            self.showNormal_raise()

    def showNormal_raise(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray.showMessage("Femify", "Still running in the tray. Right-click → Exit to quit.")

    def _poll_clip(self):
        if self.proc.poll_clip():
            self.clip_label.setText("● HOT")
            self.clip_label.setStyleSheet("color: #ff2222; font-weight: bold;")
            QTimer.singleShot(600, self._clip_ok)

    @Slot()
    def _clip_ok(self):
        self.clip_label.setText("● OK")
        self.clip_label.setStyleSheet("color: #39ff6a; font-weight: bold;")

    def _exit(self):
        self._save_now()
        self.engine.stop()
        self.tray.hide()
        self.app.quit()


def main():
    app = QApplication(sys.argv)
    apply_theme(app)
    app.setQuitOnLastWindowClosed(False)
    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.warning(None, "Femify", "System tray not available — window will stay open.")
    win = MainWindow(app)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
