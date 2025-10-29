import argparse
import json
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--submission_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    results_file = os.path.join(args.results_dir, "inference_results.json")
    summary_file = os.path.join(args.results_dir, "summary.json")

    with open(results_file, "r") as f:
        results = json.load(f)
    with open(summary_file, "r") as f:
        summary = json.load(f)

    """
    UnstableOsmosis.zip
    |_ GEN_FLUX_SMALL
    |  |_ 37FC359AB91C0DC6D21D270AED0C87E3_UnstableOsmosis_FLUX.png
    |  |_ …
    |_ GEN_SD_LARGE
    |  |_ 37FC359AB91C0DC6D21D270AED0C87E3_UnstableOsmosis_SD.png
    |  |_ …
    |_ …

    [group_name] / ["RET"|"GEN"] + _ + [approach_name] + _ + ["LARGE"|"SMALL"]/ [article_id] + _ + [group_name] + _ + [approach_name].png
    """

    print(results[0]["article_id"])
    print(summary)
    approach_name = f"{summary['model_id'].split('/')[-1]}-{summary['reranking_algorithm']}"
    approach_dir = f"RET_{approach_name}_SMALL"

    os.makedirs(os.path.join(args.submission_path, approach_dir), exist_ok=True)
    
    for result in results:
        print(result["article_id"])
        print(result["query"])
        name = f"{result['article_id']}_newsimages-um-rtl_{approach_name}.png"
        print(name)
        print(result["retrieved_images"][0])
        # Open and process the image
        from PIL import Image
        
        # Open the original image
        img = Image.open(result["retrieved_images"][0])
        
        target_width, target_height = 460, 260
        target_aspect = target_width / target_height
        
        orig_width, orig_height = img.size
        orig_aspect = orig_width / orig_height
        
        if orig_aspect < target_aspect:
            # Image is too tall (portrait), crop vertically from center
            new_height = int(orig_width / target_aspect)
            top = (orig_height - new_height) // 2
            img = img.crop((0, top, orig_width, top + new_height))
        elif orig_aspect > target_aspect:
            # Image is too wide, crop horizontally from center
            new_width = int(orig_height * target_aspect)
            left = (orig_width - new_width) // 2
            img = img.crop((left, 0, left + new_width, orig_height))
        
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        final_path = os.path.join(args.submission_path, approach_dir, name)
        img.save(final_path)
        
        # shutil.copy(result["retrieved_images"][0], os.path.join(args.submission_path, approach_dir, name))

if __name__ == "__main__":
    main()
