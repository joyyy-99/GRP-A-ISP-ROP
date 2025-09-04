import argparse, json, pandas as pd
from sklearn.model_selection import train_test_split

def main(parsed_csv, out_json, test_size=0.2, val_size=0.1, seed=42):
    df = pd.read_csv(parsed_csv)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    pats = sorted(df["patient_id"].astype(str).str.zfill(3).unique())

    train_p, test_p = train_test_split(pats, test_size=test_size, random_state=seed)
    train_p, val_p  = train_test_split(train_p, test_size=val_size/(1-test_size), random_state=seed)

    splits = {"train_patients": list(train_p), "val_patients": list(val_p), "test_patients": list(test_p)}
    with open(out_json, "w") as f: json.dump(splits, f, indent=2)
    print("Wrote", out_json)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed_csv", default="data/metadata/parsed.csv")
    ap.add_argument("--out_json", default="data/splits/split_by_patient.json")
    args = ap.parse_args()
    main(args.parsed_csv, args.out_json)
