"""Load NSL-KDD (the cleaned successor of KDD Cup 99) and map attacks to the 4 thesis attack families."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]

# Attack -> family, covering every attack name in KDDTrain+ and KDDTest+.
ATTACK_FAMILY = {
    "normal": "normal",
    # Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "apache2": "DoS", "mailbomb": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    # Probe / surveillance
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # Remote to Local
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L", "named": "R2L", "sendmail": "R2L", "snmpgetattack": "R2L",
    "snmpguess": "R2L", "worm": "R2L", "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    # User to Root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R", "ps": "U2R",
    "sqlattack": "U2R", "xterm": "U2R",
}


def download(data_dir: str | os.PathLike = "data") -> dict[str, Path]:
    """Download NSL-KDD train/test files if they are not already present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, name in FILES.items():
        path = data_dir / name
        if not path.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, path)
        paths[split] = path
    return paths


def load_split(path: str | os.PathLike) -> tuple[pd.DataFrame, pd.Series]:
    """Return features X and attack-family labels y for one NSL-KDD file."""
    df = pd.read_csv(path, header=None, names=FEATURES + ["attack", "difficulty"])
    unknown = set(df["attack"]) - set(ATTACK_FAMILY)
    if unknown:
        raise ValueError(f"Unmapped attack names: {sorted(unknown)}")
    y = df["attack"].map(ATTACK_FAMILY).rename("family")
    return df[FEATURES], y


def load(data_dir: str | os.PathLike = "data"):
    """Download if needed and return (X_train, y_train, X_test, y_test)."""
    paths = download(data_dir)
    X_tr, y_tr = load_split(paths["train"])
    X_te, y_te = load_split(paths["test"])
    return X_tr, y_tr, X_te, y_te
