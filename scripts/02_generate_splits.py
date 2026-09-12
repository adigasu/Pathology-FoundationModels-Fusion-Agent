import os
import csv
import json
import random
from collections import defaultdict, Counter

def generate_stratified_split(records, train_frac=0.70, val_frac=0.10, test_frac=0.20, seed=42):
    """
    Partitions patient records into Train, Val, and Test with strict patient-level isolation
    and multi-class stratification across all 7 Tumor Subtypes.
    """
    rng = random.Random(seed)
    
    # Group records by Tumor Subtype
    subtype_groups = defaultdict(list)
    for r in records:
        subtype_groups[r['Tumor Subtype']].append(r)
        
    train_set, val_set, test_set = [], [], []
    
    for subtype, items in sorted(subtype_groups.items()):
        # Shuffle deterministically within each subtype group
        shuffled = list(items)
        rng.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_test = max(1, round(n_total * test_frac))
        n_val = max(1, round(n_total * val_frac))
        n_train = n_total - n_test - n_val
        
        # Adjust if rounding causes boundary issues
        if n_train <= 0:
            n_train = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_test -= 1
                
        test_items = shuffled[:n_test]
        val_items = shuffled[n_test:n_test + n_val]
        train_items = shuffled[n_test + n_val:]
        
        test_set.extend(test_items)
        val_set.extend(val_items)
        train_set.extend(train_items)
        
    # Tag each record with split
    split_records = []
    for r in train_set:
        item = dict(r)
        item['split'] = 'train'
        item['seed'] = seed
        split_records.append(item)
    for r in val_set:
        item = dict(r)
        item['split'] = 'val'
        item['seed'] = seed
        split_records.append(item)
    for r in test_set:
        item = dict(r)
        item['split'] = 'test'
        item['seed'] = seed
        split_records.append(item)
        
    return split_records

def verify_and_save_splits(seeds=[42, 1337, 2026]):
    print("=" * 75)
    print("STEP 1.3: MULTI-SEED STRATIFIED PATIENT SPLIT GENERATION")
    print("=" * 75)
    
    input_csv = "artifacts/curated_patients.csv"
    if not os.path.exists(input_csv):
        print(f"ERROR: {input_csv} not found. Run scripts/01_data_audit.py first.")
        return
        
    with open(input_csv, 'r', encoding='utf-8') as f:
        records = list(csv.DictReader(f))
        
    print(f"Loaded {len(records)} curated patients from {input_csv}.")
    os.makedirs("artifacts/splits", exist_ok=True)
    
    summary = {}
    
    for seed in seeds:
        print(f"\n---> Generating Stratified Split for SEED {seed}:")
        split_records = generate_stratified_split(records, seed=seed)
        
        train_pats = set(r['patient_id'] for r in split_records if r['split'] == 'train')
        val_pats = set(r['patient_id'] for r in split_records if r['split'] == 'val')
        test_pats = set(r['patient_id'] for r in split_records if r['split'] == 'test')
        
        # 1. Zero leakage assertion
        leak_tv = train_pats & val_pats
        leak_tt = train_pats & test_pats
        leak_vt = val_pats & test_pats
        
        assert len(leak_tv) == 0, f"LEAKAGE: Train & Val share patients: {leak_tv}"
        assert len(leak_tt) == 0, f"LEAKAGE: Train & Test share patients: {leak_tt}"
        assert len(leak_vt) == 0, f"LEAKAGE: Val & Test share patients: {leak_vt}"
        print(f"    [OK] Zero-Leakage Check Passed: Train={len(train_pats)}, Val={len(val_pats)}, Test={len(test_pats)}")
        
        # 2. Subtype distribution per split
        train_subtypes = Counter(r['Tumor Subtype'] for r in split_records if r['split'] == 'train')
        val_subtypes = Counter(r['Tumor Subtype'] for r in split_records if r['split'] == 'val')
        test_subtypes = Counter(r['Tumor Subtype'] for r in split_records if r['split'] == 'test')
        
        all_subtypes = sorted(list(set(r['Tumor Subtype'] for r in split_records)))
        print(f"    {'Subtype':16s} | {'Train':5s} | {'Val':5s} | {'Test':5s} | {'Total':5s}")
        print("    " + "-" * 45)
        for st in all_subtypes:
            tr = train_subtypes.get(st, 0)
            va = val_subtypes.get(st, 0)
            te = test_subtypes.get(st, 0)
            tot = tr + va + te
            print(f"    {st:16s} | {tr:5d} | {va:5d} | {te:5d} | {tot:5d}")
            assert va > 0, f"STARVATION: {st} absent from validation set in seed {seed}!"
            assert te > 0, f"STARVATION: {st} absent from test set in seed {seed}!"
            
        # 3. Save CSV & JSON
        out_csv = f"artifacts/splits/splits_seed_{seed}.csv"
        out_json = f"artifacts/splits/splits_seed_{seed}.json"
        
        fieldnames = list(split_records[0].keys())
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_records)
            
        json_payload = {
            'seed': seed,
            'counts': {'train': len(train_pats), 'val': len(val_pats), 'test': len(test_pats), 'total': len(records)},
            'patient_splits': {r['patient_id']: {'wsi_id': r['WSI_ID'], 'split': r['split'], 'subtype': r['Tumor Subtype']} for r in split_records}
        }
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(json_payload, f, indent=2)
            
        print(f"    [Saved] CSV -> {out_csv}")
        print(f"    [Saved] JSON -> {out_json}")
        
    print("\n" + "=" * 75)
    print("STEP 1.3: SPLIT GENERATION COMPLETE & FROZEN")
    print("=" * 75)

if __name__ == "__main__":
    verify_and_save_splits()
