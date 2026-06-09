import os
import glob
import subprocess


input_dir = "/mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/broken_lcfs"
output_dir = "/mnt/beegfs/mantis/jrathi/AE_Model_Thesis/AEModel_jy_screenshots/broken_lcfs_processed"

os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))

print(f"Found {len(image_paths)} PNG files.")

for image_path in image_paths:
    filename = os.path.basename(image_path)
    stem, _ = os.path.splitext(filename)

    output_path = os.path.join(output_dir, f"{stem}_lcfs_sol.png")

    cmd = [
        "python",
        "extract_lcfs_sol.py",
        "--input", image_path,
        "--output", output_path,
        "--mode", "binary",
        "--tolerance", "65",
        "--lcfs_width", "1",
        "--dot_radius", "1",
        "--dot_step", "7",
    ]

    print(f"Processing: {filename}")
    subprocess.run(cmd, check=True)

print(f"Done. Saved outputs to: {output_dir}")
