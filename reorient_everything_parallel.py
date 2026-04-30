import os
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import tools

path = "/data/awias/periseg/saros/NIFTI_collected"


def process_file(file):
    if not file.endswith(".nii.gz"):
        return None

    try:
        file_path = os.path.join(path, file)

        file_sitk = sitk.ReadImage(file_path)

        direction = tools.get_direction_code(file_sitk)

        file_sitk_reoriented = tools.reorient_sitk(file_sitk, "LPS")

        sitk.WriteImage(file_sitk_reoriented, file_path)

        return file

    except Exception as e:
        return f"ERROR {file}: {e}"


def main():

    all_files = os.listdir(path)

    workers = max(os.cpu_count() - 1, 1)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in tqdm(
            executor.map(process_file, all_files),
            total=len(all_files),
            desc="Processing files"
        ):
            pass


if __name__ == "__main__":
    main()