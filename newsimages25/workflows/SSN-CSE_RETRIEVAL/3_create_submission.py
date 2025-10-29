from PIL import Image
import os
import json
import csv

GROUP_NAME = "SSN-CSE"            
APPROACH_NAME = "CLIP_KEYBERT_YFCC"  

RESULTS_FILE = "retrieval_results_yfcc.json"     
SOURCE_IMAGE_FOLDER = "yfcc100m_pool"            
SUBSET_FILE = r"newsimages_25_v1.0\subset.csv"   
TARGET_DIMENSIONS = (460, 260)                   

def load_subset_ids(csv_path):
    """Load article_ids from the subset CSV ."""
    subset_ids = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                subset_ids.add(row[0].strip())  
            except IndexError:
                continue
    return subset_ids


def prepare_submission(results, submission_name, allowed_ids=None):
    
    submission_folder = os.path.join(GROUP_NAME, submission_name)
    os.makedirs(submission_folder, exist_ok=True)
    print(f"\n[INFO] Preparing submission folder: {submission_folder}")

    total, missing = 0, 0
    for article_id, image_id in results.items():
        if allowed_ids and article_id not in allowed_ids:
            continue

        total += 1
        source_image_path = os.path.join(SOURCE_IMAGE_FOLDER, image_id)
        output_filename = f"{article_id}_{GROUP_NAME}_{APPROACH_NAME}.png"
        output_path = os.path.join(submission_folder, output_filename)

        if os.path.exists(source_image_path):
            try:
                with Image.open(source_image_path) as img:
                    img_resized = img.resize(TARGET_DIMENSIONS, Image.Resampling.LANCZOS)
                    img_resized.save(output_path, "PNG")
            except Exception as e:
                print(f"[WARN] Could not process {source_image_path}: {e}")
                missing += 1
        else:
            print(f"[WARN] Missing image: {source_image_path}")
            missing += 1

    print(f"[DONE] {submission_name}: {total} processed | {missing} missing")


with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    retrieval_results = json.load(f)

print(f"[INFO] Loaded {len(retrieval_results)} retrieval results.")

subset_ids = load_subset_ids(SUBSET_FILE)
print(f"[INFO] Loaded {len(subset_ids)} subset article IDs.")

#large
prepare_submission(retrieval_results, f"RET_{APPROACH_NAME}_LARGE")
#small
prepare_submission(retrieval_results, f"RET_{APPROACH_NAME}_SMALL", allowed_ids=subset_ids)
