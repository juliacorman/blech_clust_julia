import os
import sys
import shutil
import numpy as np


def _safe_read_bytes(path):
    with open(path, 'rb') as f:
        return f.read()


def combine_matching_dat_files(folders, output_dir, dtype=np.int16, suffix="_combined", verbose=True):
    os.makedirs(output_dir, exist_ok=True)

    file_map = {}
    for folder in folders:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        for name in sorted(os.listdir(folder)):
            if name.startswith('.'):
                continue
            stem, ext = os.path.splitext(name)
            if ext.lower() not in {'.dat', '.rhd'}:
                continue
            file_map.setdefault((stem, ext.lower()), []).append(os.path.join(folder, name))

    if not file_map:
        raise FileNotFoundError("No .dat/.rhd files found in the provided folders")

    matched = []
    skipped = []
    copied = []
    combined = []

    for (stem, ext), paths in sorted(file_map.items()):
        name = stem + ext
        lower_name = name.lower()

        if len(paths) < 2:
            skipped.append(name)
            continue

        # info.rhd: same config expected, so keep one copy instead of duplicating it.
        if lower_name == 'info.rhd':
            out_path = os.path.join(output_dir, 'info.rhd')
            shutil.copy2(paths[0], out_path)
            matched.append((stem, out_path, len(paths), os.path.getsize(paths[0])))
            copied.append((name, out_path))
            if verbose:
                print(f"Copied {name} from {paths[0]} -> {out_path}")
            continue

        # time.dat: it behaves like an integer counter, often resetting to 0 in later segments.
        if lower_name == 'time.dat':
            arrays = [np.fromfile(path, dtype=np.int32) for path in paths]
            offset = 0
            for i in range(1, len(arrays)):
                if arrays[i].size > 0:
                    if arrays[i][0] < arrays[i - 1][-1]:
                        offset = int(arrays[i - 1][-1]) + 1
                    arrays[i] = arrays[i].astype(np.int64) + offset
            combined_arr = np.concatenate([arr.astype(np.int64) for arr in arrays])
            out_path = os.path.join(output_dir, 'time.dat')
            combined_arr.astype(np.int64).tofile(out_path)
            matched.append((stem, out_path, len(paths), combined_arr.size))
            combined.append((name, out_path, combined_arr.size))
            if verbose:
                print(f"Combined {name}: {len(paths)} files -> {out_path}")
                print(f"  total samples: {combined_arr.size}")
            continue

        # vdd files are also raw sample streams and should be concatenated like the amp files.
        if ('vdd' in lower_name) and ext.lower() == '.dat':
            arrays = [np.fromfile(path, dtype=np.int16) for path in paths]
            combined_arr = np.concatenate(arrays)
            out_name = stem + suffix + ext
            out_path = os.path.join(output_dir, out_name)
            combined_arr.astype(np.int16).tofile(out_path)
            matched.append((stem, out_path, len(paths), combined_arr.size))
            combined.append((name, out_path, combined_arr.size))
            if verbose:
                print(f"Combined {name}: {len(paths)} files -> {out_path}")
                print(f"  total samples: {combined_arr.size}")
            continue

        # Generic raw .dat processing for Amp/DIN and any other raw sample stream.
        arrays = [np.fromfile(path, dtype=dtype) for path in paths]
        combined_arr = np.concatenate(arrays)
        out_name = stem + suffix + ext
        out_path = os.path.join(output_dir, out_name)
        combined_arr.astype(dtype).tofile(out_path)
        matched.append((stem, out_path, len(paths), combined_arr.size))
        combined.append((name, out_path, combined_arr.size))

        if verbose:
            print(f"Combined {stem}{ext}: {len(paths)} files -> {out_path}")
            print(f"  total samples: {combined_arr.size}")

    if verbose:
        print(f"\nMatched files: {len(matched)}")
        print(f"Copied files: {len(copied)}")
        print(f"Combined arrays: {len(combined)}")
        print(f"Skipped unmatched files: {len(skipped)}")
        if skipped:
            print("Skipped names:", ", ".join(skipped))

    return matched, skipped, copied, combined


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python combine_matching_DAT.py <folder1> <folder2> [<folder3> ...] <output_dir>")
        print("Example: python combine_matching_DAT.py /path/folder1 /path/folder2 /path/output")
        sys.exit(1)

    folders = sys.argv[1:-1]
    output_dir = sys.argv[-1]
    combine_matching_dat_files(folders, output_dir, dtype=np.int16, suffix="_combined")
