"""
GUI-based password authentication helper for the EUFL database.
The script uses functions from other scripts within the module.


Author: Jakub Zapletal
Version: 1.0
Date: 21/03/2026
Date modified: 21/03/2026
"""

import os
import tkinter as tk

from db_writer import verify_password
from helpers import Parameters

_LEGACY_SECRET_KEY_PATH = os.path.join(os.path.dirname(__file__), "secret.key")
_SETTINGS_YAML_PATH = os.path.join(os.path.dirname(__file__), "settings.yaml")


def _get_secret_key_path():
    appdata = os.environ.get("APPDATA")
    if appdata:
        secret_dir = os.path.join(appdata, ".eufl")
    else:
        secret_dir = os.path.join(os.path.expanduser("~"), ".eufl")
    os.makedirs(secret_dir, exist_ok=True)
    return os.path.join(secret_dir, "secret.key")


def _ask_password():
    result = {"value": None}

    win = tk.Tk()
    win.title("EUFL Database")
    win.resizable(False, False)
    win.configure(bg="#2b2b2b")

    win.update_idletasks()
    w, h = 340, 160
    x = (win.winfo_screenwidth() - w) // 2
    y = (win.winfo_screenheight() - h) // 2
    win.geometry(f"{w}x{h}+{x}+{y}")

    tk.Label(
        win,
        text="EUFL Catalogue Database",
        bg="#2b2b2b",
        fg="#ffffff",
        font=("Segoe UI", 11, "bold"),
    ).pack(pady=(18, 2))
    tk.Label(
        win,
        text="Enter password to continue:",
        bg="#2b2b2b",
        fg="#aaaaaa",
        font=("Segoe UI", 9),
    ).pack()

    entry = tk.Entry(
        win,
        show="•",
        width=28,
        font=("Segoe UI", 10),
        bg="#3c3f41",
        fg="#ffffff",
        insertbackground="#ffffff",
        relief="flat",
        bd=4,
    )
    entry.pack(pady=10)
    entry.focus_set()

    def on_ok(event=None):
        result["value"] = entry.get()
        win.destroy()

    def on_cancel():
        win.destroy()

    btn_frame = tk.Frame(win, bg="#2b2b2b")
    btn_frame.pack()
    tk.Button(
        btn_frame,
        text="OK",
        width=10,
        command=on_ok,
        bg="#4a7cc7",
        fg="#ffffff",
        relief="flat",
        font=("Segoe UI", 9),
        activebackground="#3a6ab5",
        cursor="hand2",
    ).pack(side="left", padx=6)
    tk.Button(
        btn_frame,
        text="Cancel",
        width=10,
        command=on_cancel,
        bg="#555555",
        fg="#ffffff",
        relief="flat",
        font=("Segoe UI", 9),
        activebackground="#444444",
        cursor="hand2",
    ).pack(side="left", padx=6)

    win.bind("<Return>", on_ok)
    win.mainloop()
    return result["value"]


def _should_save_secret() -> bool:
    """Return True if password persistence to secret.key is enabled in settings."""
    if not os.path.exists(_SETTINGS_YAML_PATH):
        return False
    try:
        general = Parameters.load_local_parameters(
            _SETTINGS_YAML_PATH, "general_parameters"
        )
        return bool(general.get("hash_password_to_secret", False))
    except Exception:
        return False


def get_db_password():
    """Return DB password from secret.key if present, otherwise prompt via GUI.
    The password is verified against the database before being returned.
    Returns None if no password is provided or verification fails.
    """
    secret_key_path = _get_secret_key_path()

    if not os.path.exists(secret_key_path) and os.path.exists(_LEGACY_SECRET_KEY_PATH):
        secret_key_path = _LEGACY_SECRET_KEY_PATH

    if os.path.exists(secret_key_path):
        with open(secret_key_path, "r", encoding="utf-8") as fh:
            password = fh.read().strip()
        if verify_password(password):
            return password
        print(
            f"Warning: Password from {secret_key_path} is invalid. Database writes will be skipped."
        )
        return None
    password = _ask_password()
    if not password:
        print("Warning: No password provided. Database writes will be skipped.")
        return None
    if not verify_password(password):
        print("Warning: Password is incorrect. Database writes will be skipped.")
        return None

    if _should_save_secret():
        secret_key_path = _get_secret_key_path()
        with open(secret_key_path, "w", encoding="utf-8") as fh:
            fh.write(password)

    return password
