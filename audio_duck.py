"""Windows audio session ducking control.

Opts Femify's playback session out of Windows "communication activity"
ducking, so other apps (YouTube, games) no longer dim Femify's audio.
"""
import ctypes
import os
import uuid

# COM constants
CLSCTX_ALL = 23
COINIT_APARTMENTTHREADED = 2
eRender = 0
eConsole = 0

CLSID_MMDeviceEnumerator = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
IID_IAudioSessionManager2 = "{77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F}"

ole32 = ctypes.windll.ole32


class GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32),
                ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8)]


def _make_guid(s):
    u = uuid.UUID(s)
    g = GUID()
    g.Data1 = u.time_low
    g.Data2 = u.time_mid
    g.Data3 = u.time_hi_version
    for i, b in enumerate(u.bytes[8:]):
        g.Data4[i] = b
    return g


def disable_ducking_for_pid(pid=None):
    """Set ducking preference to 'do not duck' for all audio sessions
    belonging to the given process (default: current process)."""
    if pid is None:
        pid = os.getpid()

    try:
        ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    except Exception:
        pass

    try:
        # Create MMDeviceEnumerator
        clsid = _make_guid(CLSID_MMDeviceEnumerator)
        iid_enum = _make_guid("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
        enumerator = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(ctypes.byref(clsid), None, CLSCTX_ALL,
                                     ctypes.byref(iid_enum),
                                     ctypes.byref(enumerator))
        if hr != 0 or not enumerator.value:
            return False

        # GetDefaultAudioEndpoint(eRender, eConsole) - vtable index 4
        vtable = ctypes.cast(enumerator, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        GetDefaultAudioEndpoint = ctypes.cast(
            vtable[4],
            ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)))
        device = ctypes.c_void_p()
        hr = GetDefaultAudioEndpoint(enumerator, eRender, eConsole,
                                     ctypes.byref(device))
        if hr != 0 or not device.value:
            return False

        # Activate IAudioSessionManager2 - IMMDevice::Activate, vtable index 3
        iid_asm2 = _make_guid(IID_IAudioSessionManager2)
        vtable = ctypes.cast(device, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        Activate = ctypes.cast(
            vtable[3],
            ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p,
                               ctypes.POINTER(GUID), ctypes.c_uint,
                               ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)))
        asm2 = ctypes.c_void_p()
        hr = Activate(device, ctypes.byref(iid_asm2), CLSCTX_ALL, None,
                      ctypes.byref(asm2))
        if hr != 0 or not asm2.value:
            return False

        # GetSessionEnumerator - vtable index 7
        vtable = ctypes.cast(asm2, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        GetSessionEnumerator = ctypes.cast(
            vtable[7],
            ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_void_p)))
        sess_enum = ctypes.c_void_p()
        hr = GetSessionEnumerator(asm2, ctypes.byref(sess_enum))
        if hr != 0 or not sess_enum.value:
            return False

        # GetCount - vtable index 3
        vtable = ctypes.cast(sess_enum, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        GetCount = ctypes.cast(
            vtable[3],
            ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_int)))
        count = ctypes.c_int(0)
        hr = GetCount(sess_enum, ctypes.byref(count))
        if hr != 0:
            return False

        # GetSession - vtable index 4
        GetSession = ctypes.cast(
            vtable[4],
            ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_void_p)))

        changed = False
        for i in range(count.value):
            session = ctypes.c_void_p()
            hr = GetSession(sess_enum, i, ctypes.byref(session))
            if hr != 0 or not session.value:
                continue

            vtable = ctypes.cast(session, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents

            # GetProcessId - vtable index 14 (IAudioSessionControl2)
            GetProcessId = ctypes.cast(
                vtable[14],
                ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_uint32)))
            session_pid = ctypes.c_uint32(0)
            hr = GetProcessId(session, ctypes.byref(session_pid))
            if hr != 0 or session_pid.value != pid:
                continue

            # SetDuckingPreference(TRUE) - vtable index 16
            SetDuckingPreference = ctypes.cast(
                vtable[16],
                ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, ctypes.c_int))
            hr = SetDuckingPreference(session, 1)  # TRUE = don't duck
            if hr == 0:
                changed = True

        return changed
    except Exception:
        return False
    finally:
        try:
            ole32.CoUninitialize()
        except Exception:
            pass


def disable_ducking_registry():
    """Set Windows registry to disable communication ducking globally."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Multimedia\Audio",
                             0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, "UserDuckingPreference", 0,
                          winreg.REG_DWORD, 3)  # 3 = Do nothing
        winreg.CloseKey(key)
        return True
    except Exception as e:
        print(f"registry ducking failed: {e}")
        return False
