import os

# Path to your folder
folder_path = "RET_ClipPretrained_SMALL"   # change this

old_part = "FAST"
new_part = "CVG-IBA_SEEK"

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".png"):
        if old_part in filename:
            new_name = filename.replace(old_part, new_part)
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        else:
            print(f"Skipped (no match): {filename}")

print("✅ Renaming complete.")