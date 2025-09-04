import re

PATTERN = re.compile(
    r"(?P<patient_id>\d+)_"
    r"(?P<sex>[MF])_"
    r"GA(?P<ga>\d+)_"
    r"BW(?P<bw>\d+)_"
    r"PA(?P<pa>\d+)_"
    r"DG(?P<dg>\d+)_"
    r"PF(?P<pf>\d+)_"
    r"(?P<device>[A-Za-z]+)(?P<device_version>\d+)?_"
    r"S(?P<series>\d+)_"
    r"(?P<imgnum>\d+)\.jpg$",
    re.IGNORECASE,
)

def parse_filename(name: str) -> dict:
    m = PATTERN.search(name)
    if not m:
        raise ValueError(f"Filename does not match schema: {name}")
    d = m.groupdict()
    for k in ["ga", "bw", "pa", "dg", "pf", "series", "imgnum"]:
        d[k] = int(d[k])
    d["patient_id"] = d["patient_id"].zfill(3)
    d["device"] = d["device"].upper()
    return d
