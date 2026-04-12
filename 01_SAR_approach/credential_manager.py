"""
credential_manager.py
=====================
Secure credential storage for Copernicus Dataspace (CDSE).
Adapted from HydroFloodToolkit credential_manager.

Stores two credential pairs:
  - OAuth client (client_id + client_secret) → catalogue search
  - S3 access (access_key + secret_key)       → data download (no 2FA issue)

Storage location:
  Windows : %LOCALAPPDATA%/S1LeveeDownloader/
  Linux   : ~/.config/S1LeveeDownloader/

Dependencies:
    pip install cryptography
"""

import os
import json
import tkinter as tk
from tkinter import simpledialog, messagebox
from cryptography.fernet import Fernet

APP_NAME = "S1LeveeDownloader"

if os.name == "nt":
    BASE_DIR = os.path.join(os.getenv("LOCALAPPDATA"), APP_NAME)
else:
    BASE_DIR = os.path.join(os.path.expanduser("~"), ".config", APP_NAME)

# OAuth client credentials (for catalogue search)
CREDENTIALS_OAUTH_FILE = os.path.join(BASE_DIR, "oauth_cr.json")
KEY_OAUTH_FILE         = os.path.join(BASE_DIR, "oauth_cr.key")

# S3 credentials (for download – no 2FA required)
CREDENTIALS_S3_FILE    = os.path.join(BASE_DIR, "s3_cr.json")
KEY_S3_FILE            = os.path.join(BASE_DIR, "s3_cr.key")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir():
    os.makedirs(BASE_DIR, exist_ok=True)


def _generate_key(key_file: str):
    _ensure_dir()
    key = Fernet.generate_key()
    with open(key_file, "wb") as f:
        f.write(key)


def _load_key(key_file: str) -> bytes:
    if not os.path.exists(key_file):
        _generate_key(key_file)
    with open(key_file, "rb") as f:
        return f.read()


def _encrypt(value: str, key: bytes) -> str:
    return Fernet(key).encrypt(value.encode()).decode()


def _decrypt(value: str, key: bytes) -> str:
    return Fernet(key).decrypt(value.encode()).decode()


def _save(field_a: str, field_b: str, cred_file: str, key_file: str):
    _ensure_dir()
    key = _load_key(key_file)
    with open(cred_file, "w") as f:
        json.dump({"a": _encrypt(field_a, key), "b": _encrypt(field_b, key)}, f)
    messagebox.showinfo("Saved", "Credentials have been securely stored.")


def _load(cred_file: str, key_file: str):
    if not os.path.exists(cred_file):
        return None, None
    with open(cred_file, "r") as f:
        data = json.load(f)
    key = _load_key(key_file)
    return _decrypt(data["a"], key), _decrypt(data["b"], key)


def _delete(cred_file: str, key_file: str):
    removed = False
    for path in (cred_file, key_file):
        if os.path.exists(path):
            os.remove(path)
            removed = True
    if removed:
        messagebox.showinfo("Deleted", "Credentials removed.")


# ---------------------------------------------------------------------------
# OAuth client credentials (catalogue search)
# ---------------------------------------------------------------------------

def get_oauth_credentials() -> tuple:
    """
    Returns (client_id, client_secret).
    Loads from encrypted storage; prompts via GUI if not found.
    S3 credentials page: https://shapps.dataspace.copernicus.eu/dashboard/
    """
    root = tk.Tk()
    root.withdraw()

    client_id, client_secret = _load(CREDENTIALS_OAUTH_FILE, KEY_OAUTH_FILE)

    if client_id and client_secret:
        use_stored = messagebox.askyesno(
            "OAuth credentials found",
            f"Use stored OAuth client?\nClient ID: {client_id[:20]}..."
        )
        if not use_stored:
            _delete(CREDENTIALS_OAUTH_FILE, KEY_OAUTH_FILE)
            client_id, client_secret = None, None

    if not client_id or not client_secret:
        messagebox.showinfo(
            "OAuth credentials",
            "Create an OAuth client at:\nhttps://shapps.dataspace.copernicus.eu/dashboard/\n\n"
            "Then enter Client ID and Client Secret below."
        )
        client_id     = simpledialog.askstring("OAuth", "Enter Client ID:")
        client_secret = simpledialog.askstring("OAuth", "Enter Client Secret:", show="*")

        if client_id and client_secret:
            _save(client_id, client_secret, CREDENTIALS_OAUTH_FILE, KEY_OAUTH_FILE)
        else:
            messagebox.showwarning("Failed", "Both Client ID and Client Secret are required.")
            root.destroy()
            return None, None

    root.destroy()
    return client_id, client_secret


# ---------------------------------------------------------------------------
# S3 credentials (download – no 2FA issue)
# ---------------------------------------------------------------------------

def get_s3_credentials() -> tuple:
    """
    Returns (access_key, secret_key) for CDSE S3 download.
    Loads from encrypted storage; prompts via GUI if not found.

    Generate S3 credentials at:
    https://eodata-iam.dataspace.copernicus.eu
    (Login → Generate credentials → copy Access Key + Secret Key)
    """
    root = tk.Tk()
    root.withdraw()

    access_key, secret_key = _load(CREDENTIALS_S3_FILE, KEY_S3_FILE)

    if access_key and secret_key:
        use_stored = messagebox.askyesno(
            "S3 credentials found",
            f"Use stored S3 credentials?\nAccess key: {access_key[:20]}..."
        )
        if not use_stored:
            _delete(CREDENTIALS_S3_FILE, KEY_S3_FILE)
            access_key, secret_key = None, None

    if not access_key or not secret_key:
        messagebox.showinfo(
            "S3 credentials",
            "Generate S3 credentials at:\nhttps://eodata-iam.dataspace.copernicus.eu\n\n"
            "Login → Generate credentials → copy Access Key and Secret Key."
        )
        access_key = simpledialog.askstring("S3", "Enter Access Key:")
        secret_key = simpledialog.askstring("S3", "Enter Secret Key:", show="*")

        if access_key and secret_key:
            _save(access_key, secret_key, CREDENTIALS_S3_FILE, KEY_S3_FILE)
        else:
            messagebox.showwarning("Failed", "Both Access Key and Secret Key are required.")
            root.destroy()
            return None, None

    root.destroy()
    return access_key, secret_key


# ---------------------------------------------------------------------------
# Convenience: delete all stored credentials
# ---------------------------------------------------------------------------

def delete_all_credentials():
    root = tk.Tk()
    root.withdraw()
    confirm = messagebox.askyesno(
        "Delete all credentials",
        "This will remove all stored OAuth and S3 credentials. Continue?"
    )
    if confirm:
        _delete(CREDENTIALS_OAUTH_FILE, KEY_OAUTH_FILE)
        _delete(CREDENTIALS_S3_FILE, KEY_S3_FILE)
    root.destroy()
