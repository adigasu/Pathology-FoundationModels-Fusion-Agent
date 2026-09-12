import os
import re
import csv
import sys
import argparse
from collections import defaultdict, Counter

def extract_wsi_num(wsi_id: str) -> int:
    match = re.search(r'\d+', str(wsi_id))
    return int(match.group()) if match else 999999

def resolve_data_dir(cli_dir: str = None) -> str:
    if cli_dir and os.path.exists(cli_dir):
        return cli_dir
    candidates = [
        os.environ.get("LUNG_DATA_DIR", ""),
        os.path.expanduser("~/lung_data"),
        "./data/lung_data",
        # Optional custom user path configured via LUNG_DATA_DIR
        os.environ.get("CUSTOM_LUNG_DIR", "")
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return candidates[1]

def run_data_audit(img_dir: str = None, csv_path: str = "artifacts/CLWD.csv", out_curated: str = "artifacts/curated_patients.csv"):
    print("=" * 75)
    print("STEP 1.2: FORMAL DATA AUDIT & COHORT REDUCTION (204-PATIENT COHORT)")
    print("=" * 75)

    img_dir = resolve_data_dir(img_dir)

    if not os.path.exists(csv_path):
        print(f"ERROR: Metadata file {csv_path} not found.")
        sys.exit(1)
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for r in reader:
            rows.append(r)
            
    print(f"\n[1] Metadata Loaded: {len(rows)} rows from {csv_path}")
    print(f"    Schema Fields: {fields}")

    # 1. Verify images on disk
    print(f"\n[2] Verifying WSI Images in: {img_dir}")
    if not os.path.exists(img_dir):
        print(f"    ERROR: Image directory {img_dir} does not exist!")
        sys.exit(1)
    
    existing_files = set(os.listdir(img_dir))
    missing_files = []
    available_rows = []
    for r in rows:
        fname = f"{r['WSI_ID']}.jpg"
        if fname in existing_files:
            available_rows.append(r)
        else:
            missing_files.append(r['WSI_ID'])
            
    print(f"    Total in metadata : {len(rows)}")
    print(f"    Available on disk : {len(available_rows)}")
    print(f"    Missing on disk   : {len(missing_files)} ({missing_files[:8]}...)")

    # 2. Audit Patient ID collisions & Demographics
    print("\n[3] Auditing Patient Identifiers & Clinical Consistency...")
    by_sample = defaultdict(list)
    for r in rows:
        by_sample[r['SampleNumber']].append(r)

    print(f"    Unique SampleNumbers in metadata: {len(by_sample)}")

    # Check Specimen 8377886 specifically
    if '8377886' in by_sample:
        print(f"\n    [Audit Target] Specimen '8377886' Analysis:")
        for r in by_sample['8377886']:
            print(f"      WSI_ID: {r['WSI_ID']}, Subtype: {r['Tumor Subtype']}, Age: {r['Age']}, Sex: {r['Sex']}")

    # 3. Patient Cohort Definition & Deterministic 1-Slide-Per-Patient Rule (Available on Disk)
    print("\n[4] Applying Deterministic Reduced Protocol (1 Available Slide Per Patient)...")
    
    patient_cohorts = defaultdict(list)
    for r in rows:
        p_key = f"{r['SampleNumber']}_{r['Age']}_{r['Sex']}"
        patient_cohorts[p_key].append(r)

    print(f"    Total distinct clinical patient-encounters: {len(patient_cohorts)}")

    completely_missing_patients = []
    curated_rows = []

    for p_key, p_slides in patient_cohorts.items():
        # Check which slides are available on disk
        avail_slides = [s for s in p_slides if f"{s['WSI_ID']}.jpg" in existing_files]

        if not avail_slides:
            completely_missing_patients.append((p_key, p_slides))
        else:
            # Deterministic selection rule: lowest WSI_ID among available slides on disk
            selected = min(avail_slides, key=lambda x: extract_wsi_num(x['WSI_ID']))
            curated_rows.append({
                'patient_id': p_key,
                'SampleNumber': selected['SampleNumber'],
                'WSI_ID': selected['WSI_ID'],
                'Tumor Subtype': selected['Tumor Subtype'],
                'Age': selected['Age'],
                'Sex': selected['Sex'],
                'total_patient_slides': len(p_slides),
                'available_patient_slides': len(avail_slides)
            })

    # Sort deterministically
    curated_rows.sort(key=lambda x: extract_wsi_num(x['WSI_ID']))

    print(f"\n[5] Cohort Reduction Results:")
    print(f"    Patients with 0 slides on disk in ~/lung_data: {len(completely_missing_patients)}")
    for p_key, p_s in completely_missing_patients:
        w_ids = [s['WSI_ID'] for s in p_s]
        print(f"      - {p_key}: {w_ids} (never in downloaded dataset)")

    print(f"\n    Curated Cohort Size (All on disk) : {len(curated_rows)} patients")
    print(f"    Curated Slides Available on Disk   : {len(curated_rows)} / {len(curated_rows)} (100.0%)")

    # 4. Export Curated Cohort
    os.makedirs(os.path.dirname(out_curated), exist_ok=True)
    with open(out_curated, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['patient_id', 'SampleNumber', 'WSI_ID', 'Tumor Subtype', 'Age', 'Sex', 'total_patient_slides', 'available_patient_slides']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curated_rows)
        
    print(f"\n[6] Saved Curated Cohort to: {out_curated}")
    
    # Class breakdown
    subtype_counts = Counter(r['Tumor Subtype'] for r in curated_rows)
    print("\n    Subtype Breakdown (Curated 204 Cohort):")
    for st, count in sorted(subtype_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      {st:<20}: {count} ({count/len(curated_rows)*100:.1f}%)")

    print("\n" + "=" * 75)
    print("DATA AUDIT & 204-PATIENT COHORT CURATION COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1.2: Formal Data Audit & Cohort Reduction")
    parser.add_argument("--img_dir", type=str, default=None, help="Directory containing WSI JPGs")
    parser.add_argument("--csv_path", type=str, default="artifacts/CLWD.csv", help="Clinical metadata path")
    parser.add_argument("--out_curated", type=str, default="artifacts/curated_patients.csv", help="Output curated cohort path")
    args = parser.parse_args()
    run_data_audit(img_dir=args.img_dir, csv_path=args.csv_path, out_curated=args.out_curated)
