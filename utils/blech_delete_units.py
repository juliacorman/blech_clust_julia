"""
Delete sorted units chosen by unit number (e.g. units flagged by the QA
similarity test), keeping /sorted_units and /unit_descriptor consistent.

Remaining units are renumbered 0..N-1 in their existing order. Each session is
appended to QA_output/deleted_units.txt with the reason for every deletion and
the old -> new unit numbers. Nothing downstream is re-run: spike arrays, unit
plots, QA and characteristics outputs still use the old numbering until they
are regenerated.

Optional step: it is not in the pipeline dependency graph, never writes to
execution_log.json, and no other script reads its output.

Usage:
    python utils/blech_delete_units.py <data_dir>
"""
import argparse
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import tables

from blech_clust.utils.blech_utils import entry_checker, imp_metadata

# Written next to unit_similarity_violations.txt so both QA records sit together
RECORD_NAME = 'deleted_units.txt'


def get_unit_info(hf5):
    """Return unit nodes and descriptor rows, ordered by unit number.

    Downstream code reads /unit_descriptor row i as unit i and indexes spike
    arrays by position, so numbering must be contiguous and the descriptor
    must describe the same units as /sorted_units. Anything else is refused
    before any change is made.

    Raises ValueError if /sorted_units and /unit_descriptor disagree.
    """
    if '/sorted_units' not in hf5 or '/unit_descriptor' not in hf5:
        raise ValueError('/sorted_units or /unit_descriptor not found')
    nodes = sorted(hf5.list_nodes('/sorted_units'), key=lambda n: n._v_name)
    if len(nodes) == 0:
        raise ValueError('No units found in /sorted_units')
    # Names like unit<hash> only exist mid-resort and would break int()
    if not all(re.fullmatch(r'unit\d{3}', n._v_name) for n in nodes):
        raise ValueError('Unit names must look like unit000')
    numbers = [int(n._v_name[4:]) for n in nodes]
    if numbers != list(range(len(nodes))):
        raise ValueError('Unit numbers are not contiguous from 0')

    desc = hf5.root.unit_descriptor[:]
    desc = desc[np.argsort(desc['unit_number'])]
    # The hash stored in each unit's own metadata ties a descriptor row
    # to its node, so matching hashes means the two really agree
    node_hashes = [n.unit_metadata.col('hash')[0] for n in nodes]
    if not (np.array_equal(desc['unit_number'], numbers)
            and list(desc['hash']) == node_hashes):
        raise ValueError('/unit_descriptor does not match /sorted_units')
    return nodes, desc


def desc_to_frame(desc):
    """Descriptor rows as a DataFrame, matching unit_descriptor.csv columns"""
    frame = pd.DataFrame({name: desc[name] for name in desc.dtype.names})
    # HDF5 stores the hash as bytes
    frame['hash'] = [x.decode() for x in frame['hash']]
    return frame


def delete_units(hf5, nodes, desc, delete_numbers):
    """Remove units and renumber the rest in their existing order.

    Only /sorted_units and /unit_descriptor are touched. Each unit's own
    unit_metadata table has no unit number, so it needs no update.

    Returns the new descriptor rows and the old -> new unit number table.
    """
    keep = [i for i in range(len(nodes)) if i not in delete_numbers]
    if len(keep) == 0:
        raise ValueError('Cannot delete every unit')
    for i in sorted(delete_numbers):
        nodes[i]._f_remove(recursive=True)
    # Two-step rename avoids clashes between old and new names,
    # e.g. old unit003 -> unit002 while unit002 still exists
    for old in keep:
        nodes[old]._f_rename(f'tmpunit{old:03d}')
    for new, old in enumerate(keep):
        nodes[old]._f_rename(f'unit{new:03d}')

    # Keep the surviving descriptor rows, only the unit number changes
    new_desc = desc[keep].copy()
    new_desc['unit_number'] = np.arange(len(keep))
    hf5.remove_node('/unit_descriptor')
    # Table layout is inferred from the structured array's dtype
    hf5.create_table('/', 'unit_descriptor', obj=new_desc)
    hf5.flush()
    renumbering = pd.DataFrame(
        {'old_unit_number': keep, 'new_unit_number': np.arange(len(keep))})
    return new_desc, renumbering


def write_record(record_path, hdf5_name, deleted_frame, renumbering):
    """Append one deletion session (units, reasons, renumbering) to the record"""
    # Append mode so repeated sessions on a dataset accumulate
    with open(record_path, 'a') as f:
        print(f'=== Unit deletion: {datetime.now():%Y-%m-%d %H:%M:%S} ===', file=f)
        print(f'HDF5 file: {os.path.basename(hdf5_name)}', file=f)
        print('Deleted units (unit numbers as they were before this deletion):',
              file=f)
        print(deleted_frame.to_string(index=False), file=f)
        print('', file=f)
        print('Remaining units were renumbered:', file=f)
        print(renumbering.to_string(index=False), file=f)
        print('=== End unit deletion ===', file=f)
        print('', file=f)


def ask_yes_no(msg):
    """True only for an explicit y; n and x (exit) both count as no"""
    answer, continue_bool = entry_checker(
        msg=msg,
        check_func=lambda x: x.lower() in ['y', 'n'],
        fail_response='Please enter y or n',
    )
    return continue_bool and answer.lower() == 'y'


def choose_units(unit_frame):
    """Ask for units one by one, like blech_post_process.py does for clusters.

    Returns {unit_number: reason}, which is empty if nothing was chosen,
    or None if the user typed x at the unit prompt to exit.
    """
    chosen = {}
    all_numbers = set(unit_frame['unit_number'])
    while True:
        def valid(x):
            # Blank input ends the selection
            if x == '':
                return True
            return (x.isdigit() and int(x) in all_numbers
                    and int(x) not in chosen)
        unit_str, continue_bool = entry_checker(
            msg='Unit number to delete (enter to finish): ',
            check_func=valid,
            fail_response='Enter an existing unit number not already chosen',
        )
        if not continue_bool:
            return None
        if unit_str == '':
            return chosen
        unit_num = int(unit_str)
        print(unit_frame.loc[unit_frame.unit_number == unit_num]
              .drop(columns='hash').to_string(index=False))
        # A reason is required because it goes into the deletion record
        reason, continue_bool = entry_checker(
            msg=f'Reason for deleting unit {unit_num}: ',
            check_func=lambda x: x.strip() != '',
            fail_response='Please enter a reason',
        )
        # x here only skips this unit, not the whole session
        if not continue_bool:
            print(f'Unit {unit_num} skipped')
            continue
        if ask_yes_no(f'Delete unit {unit_num}? (y/[n]): '):
            chosen[unit_num] = reason.strip()
            print(f'Unit {unit_num} marked for deletion')
        else:
            print(f'Unit {unit_num} skipped')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Delete sorted units chosen by unit number')
    parser.add_argument('data_dir', type=str,
                        help='Directory containing the data files')
    args = parser.parse_args()

    metadata_handler = imp_metadata([[], args.data_dir])
    dir_name = metadata_handler.dir_name
    os.chdir(dir_name)
    print(f'Processing : {dir_name}')

    # Same folder as unit_similarity_violations.txt
    output_dir = os.path.join(dir_name, 'QA_output')
    os.makedirs(output_dir, exist_ok=True)
    similarity_path = os.path.join(output_dir, 'unit_similarity_violations.txt')
    record_path = os.path.join(output_dir, RECORD_NAME)

    hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')
    try:
        nodes, desc = get_unit_info(hf5)
        unit_frame = desc_to_frame(desc)
        # Hash is internal and never shown to the user
        print('\n==== Units ====\n')
        print(unit_frame.drop(columns='hash').to_string(index=False))
        # Show the QA flags so units can be chosen by their flagged numbers
        if os.path.exists(similarity_path):
            print('\n==== Similarity violations (QA) ====\n')
            with open(similarity_path, 'r') as f:
                print(f.read())

        chosen = choose_units(unit_frame)
        if chosen is None or len(chosen) == 0:
            print('No units deleted')
        elif len(chosen) >= len(nodes):
            print('Cannot delete every unit, nothing changed')
        else:
            # Table for the confirmation prompt and the record file
            deleted_frame = unit_frame.loc[
                unit_frame.unit_number.isin(chosen)].drop(columns='hash')
            deleted_frame = deleted_frame.rename(
                columns={'unit_number': 'unit_number_before'})
            deleted_frame['reason'] = [
                chosen[x] for x in deleted_frame.unit_number_before]
            print('\n==== Units to delete ====\n')
            print(deleted_frame.to_string(index=False))
            # Nothing is written to the file until this final confirmation
            if ask_yes_no('Apply these changes to the HDF5 file? (y/[n]): '):
                new_desc, renumbering = delete_units(
                    hf5, nodes, desc, set(chosen))
                write_record(record_path, metadata_handler.hdf5_name,
                             deleted_frame, renumbering)
                # Keep the on-disk copy of the table in step with the HDF5 one
                csv_path = os.path.join(dir_name, 'unit_descriptor.csv')
                if os.path.exists(csv_path):
                    desc_to_frame(new_desc).to_csv(csv_path, index=False)
                print(f'\nDeleted units {sorted(chosen)}; record written to '
                      f'{record_path}')
                print('Unit numbers have changed: spike arrays, unit plots, '
                      'QA and characteristics outputs still use the old numbering')
            else:
                print('No changes made')
    except ValueError as e:
        # Raised before any change by the consistency and safety checks
        print(f'Not changing anything: {e}')
    finally:
        hf5.close()
