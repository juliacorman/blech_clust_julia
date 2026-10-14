LFP SPECTROGRAM PIPELINE - READ ME
==================================

What this does
--------------
Takes the local field potentials (LFPs) of one or more recording sessions and
produces, for every brain region and every taste:

  * a check of which LFP channels are usable,
  * artifact-free copies of the LFP trials,
  * spectrograms and power spectra for each session,
  * grand averages across sessions.

All times are aligned to stimulus (taste) delivery, which is 0 ms on every
plot. Any number of tastes and regions is supported
Your original recordings and HDF5 files are never changed.


Before you start
----------------
1. Open a terminal and activate the blech_clust conda environment
   (the one where blech_clust itself runs).
2. Go to the blech_clust folder. Run every command below from there:
       cd /path/to/blech_clust
   If you get "No module named blech_clust", the folder that CONTAINS
   blech_clust must be on your Python path:
       export PYTHONPATH=/path/to/folder_containing_blech_clust
3. Each session ("dataset") needs to be a folder that already went through
   blech_clust (blech_exp_info, blech_init, ...). It must contain:
     - one *.h5 file   (the recording; the parsed LFPs are stored in it as
                        /Parsed_LFP. If they are missing, the first script
                        you run will extract them, which needs the raw
                        amp-*.dat files and trial_info_frame.csv in the
                        folder and can take a while)
     - one *.info file (taste names, dig-in names, electrode-to-region layout)
     - one *.params file (spike_array_durations = [ms before, ms after] the
                        stimulus; this is how the plots know where 0 is)
     - trial_info_frame.csv (when the LFPs are extracted, this is what cuts
                        the recording into trials around each taste delivery)
   "Traditional" (non one-file) data formats are not supported.


Note on long commands
---------------------
Some examples below are split over several lines with a backslash (\) at the
end of the line. The backslash must be the VERY LAST character on the line
(no space after it), or the shell will not join the lines and you will get
"unrecognized arguments". If in doubt, type the whole command on one line.


Where the information comes from
--------------------------------
  Which taste / dig-in each trial belongs to ... trial_info_frame.csv, used
                                                once when the LFPs are first
                                                extracted into the .h5
  LFP data, one array per taste dig-in ........ .h5  (/Parsed_LFP)
  Taste names ................................. .info (taste_params)
  Which electrode is in which region .......... .info (electrode_layout)
  Time of stimulus delivery in each trial ..... .params (spike_array_durations)
Tastes are matched to LFP arrays by dig-in name, so different taste names,
different dig-in order, or more than four tastes all work.


Telling the scripts which datasets to use
-----------------------------------------
Option A - one or a few datasets, typed on the command line:
    --dataset /full/path/to/session_folder
    (repeat it for more sessions:  --dataset /path/a --dataset /path/b)
Option B - a list file: a plain text file with one session folder per line:
    /full/path/to/session_one
    /full/path/to/session_two
  used with   --data-list /full/path/to/my_list.txt
If you give neither, the script falls back to a default list file, which is set
for the original lab user's computer (DEFAULT_DATA_LIST at the top of
lfp_common.py). On any other computer ALWAYS give --dataset or --data-list, or
change that line.


THE STEPS - run them in this order
-----------------------------------

STEP 1  validate_LFP_data.py   (run once per session, first)
    python validate_LFP_data.py --dataset /path/to/session
    python validate_LFP_data.py --data-list /path/to/my_list.txt
  What it does: assigns each LFP channel to a region, flags bad channels
  (flat, noisy, NaN, very different from other channels in the same region),
  and compares tastes with each other (review only, nothing is removed).
  Look at the output before going on:
    <session>/LFP_channel_check/LFP_channel_validation.txt  (readable report)
    <session>/LFP_channel_check/*.png  (voltage plots; check the approved
                                        channels look sensible)
    lfp_channel_validation.csv and lfp_validation_report.json (same data)
  The approved channels are what all later steps use. If a taste is marked
  REVIEW, look at it, then decide whether to exclude it in Step 2.

STEP 2  remove_lfp_artifacts.py   (after Step 1)
    python remove_lfp_artifacts.py --dataset /path/to/session
    python remove_lfp_artifacts.py --data-list /path/to/my_list.txt
  What it does: for each taste, drops trials that look like artifacts (using
  only the approved channels) and saves the cleaned trials to
    <session>/cleaned_LFP_artifacts/
  Steps 3 and 4 need this folder, and stop with a message if it is missing.
  Optional: --mad-threshold 3   (default 3; lower removes more trials)

  EXCLUDING A TASTE from the plots (by taste name):
    python remove_lfp_artifacts.py --dataset /path/to/session_folder \
        --exclude-taste session_folder_name:Quinine
  - The part before the colon is the session FOLDER NAME (the last part of
    its path, not the whole path); after it is the taste name exactly as in
    the .info file (capital letters do not matter).
  - Repeat the option to exclude several tastes or sessions.
  - If two dig-ins share one taste name, both are excluded.
  - You can use a taste's index number instead of its name (0 is the first
    taste in the list printed at the start of the run, e.g. :1).
  - A misspelled name stops that session with a message listing its tastes.
  - The excluded taste is still cleaned and saved; it is only skipped by
    Steps 3 and 4. To change exclusions, re-run Step 2 (with the new
    --exclude-taste choices), then re-run Steps 3 and 4.

STEP 3  plot_LFP_spectrogram.py   (after Step 2)
    python plot_LFP_spectrogram.py --dataset /path/to/session
    python plot_LFP_spectrogram.py --data-list /path/to/my_list.txt \
        --output-dir /path/for/validation_json
  What it does: computes spectrograms of the cleaned trials, and for each
  region saves into <session>/LFP_analyses/ :
    <session>_<region>_STFT_spectrogram.png/.svg
         one panel per taste, 0 ms = taste delivery (red dashed line)
    <session>_<region>_power_spectrum_0-35Hz_by_taste.png/.svg
         power before (dashed) and after (solid) taste delivery, per taste
    <session>_STFT_data.pkl   the underlying numbers
  Power is z-scored against each trial's own pre-stimulus baseline.
  Also writes lfp_region_validation.json into --output-dir.

STEP 4  grand_average_LFP.py   (after Step 2; needs 2 or more sessions to be
                                useful, but works with 1)
    python grand_average_LFP.py --data-list /path/to/my_list.txt \
        --output-dir /path/for/results
  What it does: averages across all the sessions you give it and saves, in
  --output-dir, for each region:
    grand_average_<region>_power_spectrum.png/.svg
    grand_average_<region>_spectrogram.png/.svg  (one panel per taste)
  Tastes are combined across sessions BY NAME, so spell each taste the same in
  every session's .info file. Sessions with a different epoch length or
  sampling rate cannot be averaged and are skipped with a message.
  Step 3 is not required for Step 4; they are independent.


Quick example (two sessions, one taste excluded in one of them)
---------------------------------------------------------------
    cd /path/to/blech_clust
    python validate_LFP_data.py --dataset /data/ratA_day1_240101_120000
    python validate_LFP_data.py --dataset /data/ratB_day1_240102_130000
    # read each LFP_channel_validation.txt, then:
    python remove_lfp_artifacts.py \
        --dataset /data/ratA_day1_240101_120000 \
        --dataset /data/ratB_day1_240102_130000 \
        --exclude-taste ratB_day1_240102_130000:Quinine
    python plot_LFP_spectrogram.py \
        --dataset /data/ratA_day1_240101_120000 \
        --dataset /data/ratB_day1_240102_130000 \
        --output-dir /data/lfp_results
    python grand_average_LFP.py \
        --dataset /data/ratA_day1_240101_120000 \
        --dataset /data/ratB_day1_240102_130000 \
        --output-dir /data/lfp_results


Re-running
----------
  * New or changed data in a session: re-run Steps 1-4 for it.
  * Changed only which tastes are excluded: re-run Step 2, then 3 and 4.
  * Only want to remake plots: re-run Step 3 and/or 4 (Step 2 must have been
    run on that session before).
  * Outputs are overwritten. Nothing in the original recording is touched.
  * If one session in a list fails, the others still run; the failed ones are
    listed at the end of the run and the script exits with an error.


Things worth knowing
--------------------
  * The pipeline assumes the LFPs were extracted at 1000 Hz (the blech_clust
    default). A WARNING is printed if the extraction settings say otherwise.
  * If the number of samples per trial does not match the .params
    spike_array_durations, the scripts stop with an error rather than plot
    mis-aligned data. Fix spike_array_durations in the .params file so it
    matches how the LFPs were extracted.
  * Regions come from the .info file. A channel whose electrode is in no
    region, or in more than one, is not used. Regions named differently in
    different sessions can be merged for the grand average with
    REGION_ALIASES near the top of grand_average_LFP.py (by default "rf1" is
    treated as "rf").
  * Quality thresholds (what counts as a bad channel, when a taste is
    flagged for review) are constants at the top of validate_LFP_data.py.
    They are heuristics: check them against your own recordings.
  * Baseline for the plots is 2 s to 0.25 s before delivery; "post-stimulus"
    is 0 to 1.75 s after (constants in lfp_common.py). The 0.25 s gap exists
    because the 500 ms analysis window would otherwise smear the response
    into the baseline.
  * Every script prints --help:   python validate_LFP_data.py --help


What each file is
-----------------
  validate_LFP_data.py      Step 1  channel/region/taste checks and reports
  remove_lfp_artifacts.py   Step 2  artifact-free trials, taste exclusion
  plot_LFP_spectrogram.py   Step 3  per-session spectrograms and spectra
  grand_average_LFP.py      Step 4  averages across sessions
  lfp_common.py             not run directly; code and settings shared by the
                            four scripts (timing, taste names, STFT settings,
                            baseline normalisation, default paths)
  tests/test_lfp_common.py, tests/test_remove_lfp_artifacts.py
                            automated tests (run with pytest)
  LFP_validation_pipeline_flowchart.jpg/.svg
                            one-page picture of the same pipeline


Troubleshooting
---------------
  "Missing .../artifact_removal_report.json. Run remove_lfp_artifacts.py first"
      -> do Step 2 for that session before Step 3 or 4.
  "Cannot work out where stimulus delivery falls in the LFP epochs"
      -> .params spike_array_durations does not match the stored LFP length.
  "has no matching taste in the .info file"
      -> a dig-in in the HDF5 is not listed in the .info taste_params
         dig_in_names.
  "Cannot exclude taste 'X'"
      -> misspelled taste name; the message lists the available names.
  "No module named blech_clust" / "No module named easygui"
      -> wrong environment, or PYTHONPATH not set (see "Before you start").
  A session is missing from the grand average
      -> read the run log: it was either skipped (different epoch/sampling
         rate, all tastes excluded) or listed as failed at the end.
