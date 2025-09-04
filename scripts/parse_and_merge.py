import argparse, pathlib, pandas as pd
from tqdm import tqdm
from src.rop_pipeline.parse_filenames import parse_filename

def collect_metadata(images_root: pathlib.Path) -> pd.DataFrame:
    files = list(images_root.glob("*.jpg"))
    rows = []
    for p in tqdm(files, desc="Parsing filenames"):
        try:
            meta = parse_filename(p.name)
            meta["filepath"] = str(p.resolve())
            rows.append(meta)
        except Exception as e:
            rows.append({"filepath": str(p.resolve()), "error": str(e)})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", default=r"C:\Users\joyaw\OneDrive\Desktop\archive\images_stack_without_captions")   # path to images_stack_without_captions
    ap.add_argument("--patient_info", default=r"C:\Users\joyaw\OneDrive\Desktop\archive\infant_retinal_database_info.csv")  # csv or xlsx
    ap.add_argument("--out_csv", default="data/metadata/parsed.csv")
    args = ap.parse_args()

    images_root = pathlib.Path(args.images_root)
    df_files = collect_metadata(images_root)

    # load patient info
    if args.patient_info.lower().endswith(".xlsx"):
        df_info = pd.read_excel(args.patient_info)
    else:
        df_info = pd.read_csv(args.patient_info)

    # standardize patient_id in patient info
    if "patient_id" not in df_info.columns:
        df_info.rename(columns={df_info.columns[0]: "patient_id"}, inplace=True)
    df_info["patient_id"] = df_info["patient_id"].astype(str).str.zfill(3)

    df = df_files.merge(df_info, on="patient_id", how="left", suffixes=("", "_pi"))
    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    errs = df["error"].notna().sum() if "error" in df.columns else 0
    print(f"Saved: {args.out_csv} | images: {len(df)} | filename_parse_errors: {errs}")

if __name__ == "__main__":
    main()
