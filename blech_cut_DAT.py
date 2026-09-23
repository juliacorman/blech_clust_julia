import os
import numpy as np
import shutil

# Configuration parameters
def cut_DAT_files(dat_path, duration_to_keep):
    """
    dat_path:: where raw intan recording files are saved
    duration_to_keep (int):: duration of recording to keep in minutes
    only cut AMP and DIN files
    """
    files = os.listdir(dat_path)
    # make a folder to save initial long files
    long_f_folders = os.path.join(dat_path, 'long_files')
    os.makedirs(long_f_folders, exist_ok=True)
    for file_name in files:
        if ('amp' in file_name) or ('DIN' in file_name):
            print(file_name)
            input_path = os.path.join(dat_path, file_name)
            output_name = file_name.split('.')[0]+'-short'+'.dat' 
            output_path = os.path.join(dat_path, output_name)
            dtype_size = np.int16

            # 1. Calculate total samples based on file size
            file_size_bytes = os.path.getsize(input_path)
            bytes_per_sample = np.dtype(dtype_size).itemsize
            total_samples = file_size_bytes // bytes_per_sample #(num_channels * bytes_per_sample)

            # 2. Memory-map the file to avoid loading all data into RAM
            data_mm = np.memmap(
                input_path, dtype=dtype_size, mode='r', shape=(total_samples)
            )

            # 3. Define your cut window (in sample indices)
            # Example: keep from sample 10,000 to 50,000
            end_sample = 30 * 1000 * (int(duration_to_keep) * 60) # 50 min of data

            sliced_data = data_mm[:end_sample] #, :]

            # 4. Save the sliced portion to a new .dat file
            with open(output_path, 'wb') as f:
                sliced_data.tofile(f)

            # 5. Move long recorded file to the backup folder
            shutil.move(input_path, long_f_folders)

    # 6. Rename the shortened file
    files = os.listdir(dat_path)
    for f in files:
        if '-short' in f:
            re_name = f.replace("-short", "")
            old_name = os.path.join(dat_path, f)
            new_name = os.path.join(dat_path, re_name)
            print(f, re_name)
            os.rename(old_name, new_name)

if __name__ == "__main__":
    import sys
    dat_path = sys.argv[1]
    duration_to_keep = int(sys.argv[2])
    cut_DAT_files(dat_path, duration_to_keep)

