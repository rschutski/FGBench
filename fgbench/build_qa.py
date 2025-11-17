from tqdm.auto import tqdm
from rdkit import DataStructs
import pandas as pd
from rdkit import Chem
from collections import Counter, defaultdict
import numpy as np
import argparse
import ast
import logging
from typing import Optional, Sequence, List, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)

from accfg import (AccFG, draw_mol_with_fgs, molimg, set_atom_idx,
                   img_grid, compare_mols, draw_compare_mols,
                   draw_RascalMCES, print_fg_tree, remove_atoms_from_mol,
                   SmilesMCStoGridImage, remove_fg_list_from_mol, get_RascalMCES,
                   remove_atoms_add_hs, get_outer_bond_from_fg_list)
afg_lite = AccFG(print_load_info=True, lite=True)

from .prompts.question import (single_bool_classification_question, single_bool_regression_question, single_value_regression_question, 
                               interaction_bool_classification_question, interaction_bool_regression_question, interaction_value_regression_question, 
                               comparison_bool_classification_question, comparison_bool_regression_question, comparison_value_regression_question)


def merge_diff_tuple(diff_tuple_list: List[Tuple[Any, ...]]) -> List[Any]:
    """Merge a list of functional group difference tuples into a single flat list.
    
    Args:
        diff_tuple_list: List of tuples
    
    Returns:
        A flat list containing all elements from all tuples in the input list.
    """
    merged_diff = []
    for diff_tuple in diff_tuple_list:
        merged_diff += diff_tuple
    return merged_diff


def exam_comparison(target_smiles: str, ref_smiles: str, target_diff: List[Tuple[Any, ...]], ref_diff: List[Tuple[Any, ...]]) -> bool:
    """Examine whether two molecules share the same scaffold after removing functional groups.
    
    This function compares a target molecule and a reference molecule by removing
    their respective functional group differences and checking if the remaining
    scaffolds (core structures) are identical. This is used to verify that two
    molecules can be meaningfully compared based on their functional group differences.
    
    Args:
        target_smiles: SMILES string of the target molecule to compare.
        ref_smiles: SMILES string of the reference molecule to compare against.
        target_diff: List of tuples representing functional groups to remove from
            the target molecule. Each tuple contains (fg_name, fg_smiles, fg_atoms, ...).
        ref_diff: List of tuples representing functional groups to remove from
            the reference molecule. Each tuple contains (fg_name, fg_smiles, fg_atoms, ...).
    
    Returns:
        bool: True if the remaining scaffolds after removing functional groups are
            identical (canonical SMILES match), False otherwise.
    """
    target_mol = Chem.MolFromSmiles(target_smiles)
    ref_mol = Chem.MolFromSmiles(ref_smiles)

    target_mol = set_atom_idx(target_mol,'atomNote')
    ref_mol = set_atom_idx(ref_mol, 'atomNote')
    target_fg_diff = merge_diff_tuple(target_diff)
    ref_fg_diff = merge_diff_tuple(ref_diff)
    target_remain_mol = remove_fg_list_from_mol(target_mol, target_fg_diff)
    ref_remain_mol = remove_fg_list_from_mol(ref_mol, ref_fg_diff)
    Chem.SanitizeMol(target_remain_mol)
    Chem.SanitizeMol(ref_remain_mol)
    target_remain_smi = Chem.MolToSmiles(target_remain_mol, isomericSmiles=False)
    ref_remain_smi = Chem.MolToSmiles(ref_remain_mol, isomericSmiles=False)
    logger.debug(f'Target remaining scaffold: {target_remain_smi}')
    logger.debug(f'Reference remaining scaffold: {ref_remain_smi}')
    return target_remain_smi == ref_remain_smi


def sort_bond_tuple(bond_tuple: Tuple[int, ...]) -> Tuple[int, ...]:
    """Sort the atoms in a bond tuple to create a canonical representation.
    
    Args:
        bond_tuple: Tuple containing atom indices representing a bond, typically
            (atom_idx1, atom_idx2) or similar structure.
    
    Returns:
        tuple: A sorted tuple of atom indices
    """
    return tuple(sorted(bond_tuple)) # type: ignore


def get_frag_name_smi_from_atom(fg_tuple_list: List[Tuple[str, str, Any, Any, Any]], atom_idx: int) -> Optional[str]:
    """Retrieve the functional group name and SMILES string containing a specific atom.
    
    This function searches through a list of functional group tuples to find which
    functional group contains a given atom index. It returns a formatted string
    containing the functional group name and its SMILES representation.
    
    Args:
        fg_tuple_list: List of tuples, where each tuple contains functional group
            information in the format (fg_name, fg_smiles, fg_atoms, outer_bonds, root_bond).
        atom_idx: Integer index of the atom to search for within the functional groups.
    
    Returns:
        str: Formatted string containing the functional group name and SMILES in the
            format "fg_name (fg_smiles)", or None if the atom is not found in any
            functional group.
    """
    for fg_name, fg_smiles, fg_atoms, outer_bonds, root_bond in fg_tuple_list:
        if atom_idx in fg_atoms:
            return f'{fg_name} ({fg_smiles})'
    return None


def rebuild_from_comparison(target_smiles: str, ref_smiles: str, target_diff: List[Tuple[Any, ...]], ref_diff: List[Tuple[Any, ...]]) -> Optional[Dict[str, Any]]:
    """Rebuild molecular comparison information by analyzing functional group differences.
    
    This function performs a comprehensive analysis of two molecules (target and reference)
    by identifying their common scaffold and extracting detailed information about the
    functional group differences. It creates atom mappings between the molecules, identifies
    which functional groups need to be disconnected and connected, and generates a structured
    representation of the edit plan needed to transform the reference molecule into the
    target molecule.
    
    The function validates that both molecules share the same scaffold after removing
    their respective functional group differences, then uses maximum common edge subgraph
    (MCES) matching to establish atom correspondences. It identifies outer bonds of
    functional groups and creates connection plans for adding new functional groups.
    
    Args:
        target_smiles: SMILES string of the target molecule.
        ref_smiles: SMILES string of the reference molecule.
        target_diff: List of tuples representing functional groups present in target
            but not in reference. Each tuple contains (fg_name, fg_smiles, fg_atoms, ...).
        ref_diff: List of tuples representing functional groups present in reference
            but not in target. Each tuple contains (fg_name, fg_smiles, fg_atoms, ...).
    
    Returns:
        dict: Dictionary containing comprehensive comparison information with keys:
            - 'target_smiles': Original target SMILES string
            - 'target_mapped_smiles': Target SMILES with atom mapping numbers
            - 'ref_smiles': Original reference SMILES string
            - 'ref_mapped_smiles': Reference SMILES with atom mapping numbers
            - 'target_diff': Original target_diff input
            - 'ref_diff': Original ref_diff input
            - 'disconnect_list': List of tuples (fg_name, fg_atoms) indicating
                functional groups to remove from target
            - 'connect_dict': Dictionary mapping functional group names to lists of
                connection tuples (in_atom, out_atom, out_frag) indicating how to
                add functional groups
        Returns None if the molecules do not share the same scaffold after removing
        functional group differences.
    """
    target_mol = Chem.MolFromSmiles(target_smiles)
    ref_mol = Chem.MolFromSmiles(ref_smiles)

    target_mol = set_atom_idx(target_mol,'atomNote')
    ref_mol = set_atom_idx(ref_mol, 'atomNote')
    
    target_mol_with_mapped_atom = Chem.MolFromSmiles(target_smiles)
    target_mol_with_mapped_atom = set_atom_idx(target_mol_with_mapped_atom, 'molAtomMapNumber')
    target_mapped_smiles = Chem.MolToSmiles(target_mol_with_mapped_atom)
    
    ref_mol_with_mapped_atom = Chem.MolFromSmiles(ref_smiles)
    ref_mol_with_mapped_atom = set_atom_idx(ref_mol_with_mapped_atom, 'molAtomMapNumber')
    ref_mapped_smiles = Chem.MolToSmiles(ref_mol_with_mapped_atom)
    
    target_fg_diff = merge_diff_tuple(target_diff)
    disconnect_list = []
    for fg_name, _, fg_atoms in target_fg_diff:
        disconnect_list.append((fg_name, fg_atoms))
    
    ref_fg_diff = merge_diff_tuple(ref_diff)
    target_remain_mol = remove_fg_list_from_mol(target_mol, target_fg_diff)
    ref_remain_mol = remove_fg_list_from_mol(ref_mol, ref_fg_diff)
    Chem.SanitizeMol(target_remain_mol)
    Chem.SanitizeMol(ref_remain_mol)

    if Chem.MolToSmiles(target_remain_mol, isomericSmiles=False) != Chem.MolToSmiles(ref_remain_mol, isomericSmiles=False):
        return None
    res_on_common_structure = get_RascalMCES(target_remain_mol, ref_remain_mol)
    atom_matches = res_on_common_structure[0].atomMatches()
    atom_matches_on_note = []
    for taget_idx, ref_idx in atom_matches:
        target_note = int(target_remain_mol.GetAtomWithIdx(taget_idx).GetProp('atomNote'))
        ref_note = int(ref_remain_mol.GetAtomWithIdx(ref_idx).GetProp('atomNote'))
        atom_matches_on_note.append((target_note, ref_note))
    target_to_ref_note_map = dict(atom_matches_on_note)
    ref_to_target_note_map = {ref_note: target_note for target_note, ref_note in atom_matches_on_note}
    ref_note_mapped_list = list(ref_to_target_note_map.keys())
    
    # Get outer bonds of each fg in ref molecule
    ref_fg_list_outer_bonds = get_outer_bond_from_fg_list(ref_mol, ref_fg_diff)

    fg_tuple_list = []
    unique_bonds = []
    unique_fg_smi = []
    for fg_name, n, fg_list, outer_bonds_list in ref_fg_list_outer_bonds:
        for fg_atoms, outer_bonds in zip(fg_list, outer_bonds_list):
            fg_smiles = Chem.MolFragmentToSmiles(ref_mol_with_mapped_atom,fg_atoms)
            unique_fg_smi.append(fg_smiles)
            root_bond = []
            for outer_bond in outer_bonds:
                outer_atom = outer_bond[1]
                if outer_atom in ref_note_mapped_list:
                    root_bond.append(outer_bond)
                unique_bonds.append(sort_bond_tuple(outer_bond))
            fg_tuple_list.append((fg_name, fg_smiles, fg_atoms, outer_bonds, root_bond))
    unique_bonds = set(unique_bonds)
    unique_fg_smi = set(unique_fg_smi)
    fg_tuple_list = sorted(fg_tuple_list, key=lambda x: x[-1], reverse=True)
    
    connect_list = []
    connect_dict = defaultdict(list)
    for fg_name, fg_smiles, fg_atoms, outer_bonds, root_bond in fg_tuple_list:
        if fg_smiles in unique_fg_smi:
            unique_fg_smi.discard(fg_smiles)
            for outer_bond in outer_bonds:
                if sort_bond_tuple(outer_bond) in unique_bonds:
                    unique_bonds.discard(sort_bond_tuple(outer_bond))
                    in_atom = outer_bond[0]
                    out_atom = outer_bond[1]
                    if out_atom in ref_note_mapped_list:
                        out_atom = ref_to_target_note_map[out_atom]
                        out_frag = 'target molecule'
                    else:
                        out_frag = get_frag_name_smi_from_atom(fg_tuple_list, out_atom)
                    connect_list.append((fg_name, fg_smiles, in_atom, out_atom, out_frag))
                    connect_dict[f'{fg_name} ({fg_smiles})'].append((in_atom, out_atom, out_frag))

    result = {'target_smiles': target_smiles,
              'target_mapped_smiles': target_mapped_smiles,
              'ref_smiles': ref_smiles,
              'ref_mapped_smiles': ref_mapped_smiles,
              'target_diff': target_diff,
              'ref_diff': ref_diff,
              'disconnect_list': disconnect_list,
              'connect_dict': connect_dict}
    return result


def load_dataset(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a molecular dataset and its comparison data from CSV files.
    
    This function loads two CSV files: one containing molecular SMILES and property
    labels, and another containing pairwise molecular comparisons with functional
    group differences. The comparison DataFrame's diff columns are parsed from
    string representations to Python objects (lists/tuples).
    
    Args:
        dataset_name: String identifier for the dataset (e.g., 'esol', 'hiv').
            Used to construct file paths as 'data/molnet/{dataset_name}.csv' and
            'data/molnet/{dataset_name}_compare.csv'.
    
    Returns:
        tuple: A tuple containing two DataFrames:
            - df: DataFrame with columns including 'smiles' and task-specific
                property columns (named as '0', '1', etc.)
            - compare_df: DataFrame with columns 'target_smiles', 'ref_smiles',
                'target_diff', and 'ref_diff', where diff columns are parsed
                from string representations to Python objects
    """
    df = pd.read_csv(f'data/molnet/{dataset_name}.csv')
    compare_df = pd.read_csv(f'data/molnet/{dataset_name}_compare.csv')
    compare_df['target_diff'] = compare_df['target_diff'].apply(lambda x: eval(x))
    compare_df['ref_diff'] = compare_df['ref_diff'].apply(lambda x: eval(x))
    return df, compare_df


def build_edit_text(disconnect_list: List[Tuple[str, Any]], connect_dict: Dict[str, List[Tuple[int, int, str]]]) -> str:
    """Build a human-readable text description of molecular edit plan.
    
    This function creates a natural language description of the functional group
    modifications needed to transform one molecule into another. It describes
    which functional groups are being removed and which are being added, along
    with their positions and connection details.
    
    Args:
        disconnect_list: List of tuples (fg_name, fg_atoms) representing functional
            groups to be removed from the molecule. Each tuple contains the functional
            group name and the atom indices where it is located.
        connect_dict: Dictionary mapping functional group names (as strings in format
            "fg_name (fg_smiles)") to lists of connection tuples. Each connection
            tuple contains (in_atom, out_atom, out_frag) describing how to connect
            the functional group: in_atom is the atom in the FG to connect, out_atom
            is the target atom, and out_frag describes where out_atom belongs.
    
    Returns:
        str: A formatted text string describing the edits, with sections for:
            - Removing functional groups (if disconnect_list is non-empty)
            - Adding functional groups (if connect_dict is non-empty)
            Returns an empty string if both inputs are empty (though this case
            should not occur in practice).
    """
    if disconnect_list and connect_dict:
        removed_fgs = '\n'.join([f'* removing {fg_name} at position {fg_atoms}' for fg_name, fg_atoms in disconnect_list])
        added_fgs_list = []
        for fg_name,atom_change_list in connect_dict.items():
            added_text = f'* adding {fg_name} by ' + ', '.join([f'connecting its position {in_atom} to the position {out_atom} of {out_frag}' for in_atom, out_atom, out_frag in atom_change_list])
            added_fgs_list.append(added_text)
        added_fgs = '\n'.join(added_fgs_list)
        return f'by removing the following functional groups: \n{removed_fgs} \nand adding the following functional groups: \n{added_fgs}'
    
    if disconnect_list and not connect_dict:
        removed_fgs = '\n'.join([f'* removing {fg_name} at position {fg_atoms}' for fg_name, fg_atoms in disconnect_list])
        return f'by removing the following functional groups: \n{removed_fgs}'
    
    if not disconnect_list and connect_dict:
        added_fgs_list = []
        for fg_name,atom_change_list in connect_dict.items():
            added_text = f'* adding {fg_name} by ' + ', '.join([f'connecting its position {in_atom} to the position {out_atom} of {out_frag}' for in_atom, out_atom, out_frag in atom_change_list])
            added_fgs_list.append(added_text)
        added_fgs = '\n'.join(added_fgs_list)
        return f'by adding the following functional groups: \n{added_fgs}'


def filter_one_fg(row: Union[pd.Series, Dict[str, Any]]) -> bool:
    """Determine if a row represents a single functional group difference.
    
    This function checks whether a comparison row represents a case where exactly
    one functional group differs between the target and reference molecules. This
    is used to distinguish between single functional group edits (simpler cases)
    and multi-functional group interactions (more complex cases that may involve
    synergistic effects).
    
    A single functional group difference is defined as:
    - Exactly one functional group in target_diff and none in ref_diff, OR
    - Exactly one functional group in ref_diff and none in target_diff
    
    Args:
        row: pandas Series or dictionary-like object containing comparison information.
            Must have keys 'target_diff' and 'ref_diff', where each diff is a list
            of tuples representing functional group differences.
    
    Returns:
        bool: True if the row represents exactly one functional group difference
            (single FG edit), False if it represents zero or multiple functional
            group differences (interaction case).
    """
    target_diff = row['target_diff']
    ref_diff = row['ref_diff']
    target_fg_diff = merge_diff_tuple(target_diff)
    ref_fg_diff = merge_diff_tuple(ref_diff)
    empty_target_diff = len(target_fg_diff) == 0
    empty_ref_diff = len(ref_fg_diff) == 0
    one_target_fg = len(target_fg_diff) == 1
    one_ref_fg = len(ref_fg_diff) == 1
    only_one_fg = (one_target_fg and empty_ref_diff) or (one_ref_fg and empty_target_diff)
    return only_one_fg


def build_qa_from_dataframes(
    smiles_property_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    task_list: Sequence[str],
    tag: str,
    output_file: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """Build QA pairs from in-memory DataFrames
    1) reconstructs edit plans between target/reference molecules,
    2) attaches property labels for each task,
    3) generates task- and edit-type-specific questions/answers,
    4) concatenates all QAs across tasks into a single DataFrame,
    5) optionally writes the consolidated QA DataFrame to ``output_file`` in JSONL format.

    The expected schema is:
    - smiles_property_df: must contain a ``'smiles'`` column and one column per task,
      where task columns are named as stringified indices: ``'0'``, ``'1'``, ...
    - compare_df: must contain ``'target_smiles'``, ``'ref_smiles'``, ``'target_diff'``, ``'ref_diff'``.
      The diff columns may be python-literal strings (e.g. saved via ``str(list)``) or already-evaluated objects.

    Args:
        smiles_property_df: DataFrame with molecule SMILES and per-task labels.
            Required columns: ``'smiles'``, plus one column per task named ``str(task_index)``.
        compare_df: DataFrame with molecule pairs and structural differences.
            Required columns: ``'target_smiles'``, ``'ref_smiles'``, ``'target_diff'``, ``'ref_diff'``.
        task_list: Ordered property names. Index in this sequence maps to column names ``'0'``, ``'1'``, ...
        tag: Either ``'classification'`` or ``'regression'`` to control QA generation.
        output_file: Optional path to write the consolidated QA JSONL (``orient='records', lines=True``).
        dataset_name: Optional dataset identifier to populate the ``'dataset'`` column. Defaults to ``'custom'``.

    Returns:
        A DataFrame containing the concatenated QAs across tasks. Columns include:
        - ``question`` (str), ``answer`` (str or float),
        - metadata copied from comparison info (e.g., ``target_smiles``, ``ref_smiles``, diffs, edit plan),
        - ``type`` (QA subtype), ``dataset`` (``dataset_name`` or ``'custom'``), and ``task_num`` (int).

    Raises:
        ValueError: If ``tag`` is not one of ``'classification'`` or ``'regression'``.
    """
    if tag not in ('classification', 'regression'):
        raise ValueError("tag must be 'classification' or 'regression'")

    dataset_tag = dataset_name if dataset_name is not None else 'custom'

    # Normalize diff columns: if stored as strings, safely parse to python objects.
    def _maybe_literal_eval(series: pd.Series) -> pd.Series:
        def _parse(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    # If parsing fails, leave as-is and let downstream raise if invalid
                    return x
            return x
        return series.apply(_parse)

    if 'target_diff' in compare_df.columns:
        compare_df = compare_df.copy()
        compare_df['target_diff'] = _maybe_literal_eval(compare_df['target_diff'])
    if 'ref_diff' in compare_df.columns:
        compare_df['ref_diff'] = _maybe_literal_eval(compare_df['ref_diff'])

    # Stage 1: build comparison info with labels per task.
    compare_info_dict_list = []
    for i in tqdm(range(len(compare_df))):
        target_smiles = compare_df.loc[i, 'target_smiles']
        ref_smiles = compare_df.loc[i, 'ref_smiles']
        target_diff = compare_df.loc[i, 'target_diff']
        ref_diff = compare_df.loc[i, 'ref_diff']

        # Compute edit plan on common scaffold; skip failures.
        try:
            compare_info_dict = rebuild_from_comparison(target_smiles, ref_smiles, target_diff, ref_diff)
        except Exception:
            compare_info_dict = None
        if compare_info_dict is None:
            continue

        # For each task, attach labels and property name, producing one row per task.
        for task_num, task in enumerate(task_list):
            # Expect label columns to be named '0', '1', ... matching enumerate(task_list)
            col_name = str(task_num)
            try:
                target_rows = smiles_property_df[smiles_property_df['smiles'] == target_smiles]
                ref_rows = smiles_property_df[smiles_property_df['smiles'] == ref_smiles]
                if len(target_rows) == 0 or len(ref_rows) == 0:
                    # If either label is missing, skip this pairing for this task.
                    continue
                target_label = target_rows.iloc[0][col_name]
                ref_label = ref_rows.iloc[0][col_name]
            except Exception:
                # Any lookup/column errors -> skip this entry for robustness.
                continue

            new_compare_info_dict = compare_info_dict.copy()
            new_compare_info_dict['target_label'] = target_label
            new_compare_info_dict['ref_label'] = ref_label
            new_compare_info_dict['property_name'] = task
            compare_info_dict_list.append(new_compare_info_dict)

    compare_info_df = pd.DataFrame(compare_info_dict_list)

    # Stage 2: build QA across tasks, split by edit complexity (single vs. interaction).
    qa_df = pd.DataFrame()

    for task_num, task in enumerate(task_list):
        task_df = compare_info_df.loc[compare_info_df['property_name'] == task]
        # Split into single-FG and interaction-FG edits.
        single_fg_df = task_df[task_df.apply(filter_one_fg, axis=1)]
        interaction_fg_df = task_df[~task_df.apply(filter_one_fg, axis=1)]

        if tag == 'classification':
            sbc_list = []
            ibc_list = []
            cbc_list = []

            # Single-FG classification QAs
            for i in tqdm(range(len(single_fg_df))):
                row = single_fg_df.iloc[i]
                edit_text = build_edit_text(row['disconnect_list'], row['connect_dict'])
                sbc_question = single_bool_classification_question.format(
                    target_mapped_smiles=row['target_mapped_smiles'],
                    property_name=row['property_name'],
                    target_label='True' if row['target_label'] == 1 else 'False',
                    edit_text=edit_text
                )
                sbc_answer = 'True' if row['target_label'] != row['ref_label'] else 'False'
                sbc_list.append({'question': sbc_question, 'answer': sbc_answer} | row.to_dict())

            # Interaction-FG classification QAs
            for i in tqdm(range(len(interaction_fg_df))):
                row = interaction_fg_df.iloc[i]
                edit_text = build_edit_text(row['disconnect_list'], row['connect_dict'])
                ibc_question = interaction_bool_classification_question.format(
                    target_mapped_smiles=row['target_mapped_smiles'],
                    property_name=row['property_name'],
                    target_label='True' if row['target_label'] == 1 else 'False',
                    edit_text=edit_text
                )
                ibc_answer = 'True' if row['target_label'] != row['ref_label'] else 'False'
                ibc_list.append({'question': ibc_question, 'answer': ibc_answer} | row.to_dict())

            # Comparison classification QAs
            for i in tqdm(range(len(task_df))):
                row = task_df.iloc[i]
                _ = build_edit_text(row['disconnect_list'], row['connect_dict'])  # not used in comparison prompt
                cbc_question = comparison_bool_classification_question.format(
                    target_smiles=row['target_smiles'],
                    ref_smiles=row['ref_smiles'],
                    property_name=row['property_name'],
                    ref_label='True' if row['ref_label'] == 1 else 'False'
                )
                cbc_answer = 'True' if row['target_label'] != row['ref_label'] else 'False'
                cbc_list.append({'question': cbc_question, 'answer': cbc_answer} | row.to_dict())

            sbc_df = pd.DataFrame(sbc_list)
            ibc_df = pd.DataFrame(ibc_list)
            cbc_df = pd.DataFrame(cbc_list)

            subtask_qa_df = pd.concat([
                sbc_df.assign(type='single_bool_classification'),
                ibc_df.assign(type='interaction_bool_classification'),
                cbc_df.assign(type='comparison_bool_classification')
            ])
            subtask_qa_df['dataset'] = dataset_tag
            subtask_qa_df['task_num'] = task_num
            qa_df = pd.concat([qa_df, subtask_qa_df], ignore_index=True)

        elif tag == 'regression':
            ibr_list = []
            ivr_list = []
            cbr_list = []
            cvr_list = []
            sbr_list = []
            svr_list = []

            # Single-FG regression QAs (bool + value)
            for i in tqdm(range(len(single_fg_df))):
                row = single_fg_df.iloc[i]
                edit_text = build_edit_text(row['disconnect_list'], row['connect_dict'])
                sbr_question = single_bool_regression_question.format(
                    target_mapped_smiles=row['target_mapped_smiles'],
                    property_name=row['property_name'],
                    target_label=round(row['target_label'], 3),
                    edit_text=edit_text
                )
                sbr_answer = 'True' if row['target_label'] < row['ref_label'] else 'False'
                sbr_list.append({'question': sbr_question, 'answer': sbr_answer} | row.to_dict())

                svr_question = single_value_regression_question.format(
                    target_mapped_smiles=row['target_mapped_smiles'],
                    property_name=row['property_name'],
                    target_label=round(row['target_label'], 3),
                    edit_text=edit_text
                )
                svr_answer = row["ref_label"] - row["target_label"]
                svr_list.append({'question': svr_question, 'answer': svr_answer} | row.to_dict())

            # Interaction-FG regression QAs (bool + value)
            for i in tqdm(range(len(interaction_fg_df))):
                row = interaction_fg_df.iloc[i]
                edit_text = build_edit_text(row['disconnect_list'], row['connect_dict'])
                ibr_question = interaction_bool_regression_question.format(
                    target_mapped_smiles=row['target_mapped_smiles'],
                    property_name=row['property_name'],
                    target_label=round(row['target_label'], 3),
                    edit_text=edit_text
                )
                ibr_answer = 'True' if row['target_label'] < row['ref_label'] else 'False'
                ibr_list.append({'question': ibr_question, 'answer': ibr_answer} | row.to_dict())

                ivr_question = interaction_value_regression_question.format(
                    target_mapped_smiles=row['target_mapped_smiles'],
                    property_name=row['property_name'],
                    target_label=round(row['target_label'], 3),
                    edit_text=edit_text
                )
                ivr_answer = row["ref_label"] - row["target_label"]
                ivr_list.append({'question': ivr_question, 'answer': ivr_answer} | row.to_dict())

            # Comparison regression QAs (bool + value)
            for i in tqdm(range(len(task_df))):
                row = task_df.iloc[i]
                _ = build_edit_text(row['disconnect_list'], row['connect_dict'])  # not used in comparison prompt
                cbr_question = comparison_bool_regression_question.format(
                    target_smiles=row['target_smiles'],
                    ref_smiles=row['ref_smiles'],
                    property_name=row['property_name'],
                    ref_label=round(row['ref_label'], 3)
                )
                cbr_answer = 'True' if row['target_label'] > row['ref_label'] else 'False'
                cbr_list.append({'question': cbr_question, 'answer': cbr_answer} | row.to_dict())

                cvr_question = comparison_value_regression_question.format(
                    target_smiles=row['target_smiles'],
                    ref_smiles=row['ref_smiles'],
                    property_name=row['property_name'],
                    ref_label=round(row['ref_label'], 3)
                )
                cvr_answer = row["target_label"] - row["ref_label"]
                cvr_list.append({'question': cvr_question, 'answer': cvr_answer} | row.to_dict())

            ibr_df = pd.DataFrame(ibr_list)
            ivr_df = pd.DataFrame(ivr_list)
            cbr_df = pd.DataFrame(cbr_list)
            cvr_df = pd.DataFrame(cvr_list)
            sbr_df = pd.DataFrame(sbr_list)
            svr_df = pd.DataFrame(svr_list)

            subtask_qa_df = pd.concat([
                sbr_df.assign(type='single_bool_regression'),
                svr_df.assign(type='single_value_regression'),
                ibr_df.assign(type='interaction_bool_regression'),
                ivr_df.assign(type='interaction_value_regression'),
                cbr_df.assign(type='comparison_bool_regression'),
                cvr_df.assign(type='comparison_value_regression')
            ])
            subtask_qa_df['dataset'] = dataset_tag
            subtask_qa_df['task_num'] = task_num
            qa_df = pd.concat([qa_df, subtask_qa_df], ignore_index=True)

    # Optionally write consolidated QA DataFrame to JSONL.
    if output_file:
        qa_df.to_json(output_file, orient='records', lines=True)

    return qa_df

regression_dataset_dict = {
    'esol':['log-scale water solubility in mols per litre'],
    'lipo':['octanol/water distribution coefficient (logD at pH 7.4)'],
    'freesolv':['hydration free energy in water'],
    'qm9':[
            'Dipole moment (unit: D)',
            'Isotropic polarizability (unit: Bohr^3)',
            'Highest occupied molecular orbital energy (unit: Hartree)',
            'Lowest unoccupied molecular orbital energy (unit: Hartree)',
            'Gap between HOMO and LUMO (unit: Hartree)',
            'Electronic spatial extent (unit: Bohr^2)',
            'Zero point vibrational energy (unit: Hartree)',
            'Heat capavity at 298.15K (unit: cal/(mol*K))',
            'Internal energy at 0K (unit: Hartree)',
            'Internal energy at 298.15K (unit: Hartree)',
            'Enthalpy at 298.15K (unit: Hartree)',
            'Free energy at 298.15K (unit: Hartree)'
            ] #12
}

classification_dataset_dict = {
    # Biophysics
    'hiv':['HIV inhibitory activity'], #1
    'bace': ['human β-secretase 1 (BACE-1) inhibitory activity'], #1
    # Physiology
    'bbbp': ['blood-brain barrier penetration'], #1
    'tox21': [
                "Androgen receptor pathway activation",
                "Androgen receptor ligand-binding domain activation",
                "Aryl hydrocarbon receptor activation",
                "Inhibition of aromatase enzyme",
                "Estrogen receptor pathway activation",
                "Estrogen receptor ligand-binding domain activation",
                "Activation of peroxisome proliferator-activated receptor gamma",
                "Activation of antioxidant response element signaling",
                "Activation of ATAD5-mediated DNA damage response",
                "Activation of heat shock factor response element signaling",
                "Disruption of mitochondrial membrane potential",
                "Activation of p53 tumor suppressor pathway"
            ], #12
    'sider': [
                "Cause liver and bile system disorders",
                "Cause metabolic and nutritional disorders",
                "Cause product-related issues",
                "Cause eye disorders",
                "Cause abnormal medical test results",
                "Cause muscle, bone, and connective tissue disorders",
                "Cause gastrointestinal disorders",
                "Cause adverse social circumstances",
                "Cause immune system disorders",
                "Cause reproductive system and breast disorders",
                "Cause tumors and abnormal growths (benign, malignant, or unspecified)",
                "Cause general disorders and administration site conditions",
                "Cause endocrine (hormonal) disorders",
                "Cause complications from surgical and medical procedures",
                "Cause vascular (blood vessel) disorders",
                "Cause blood and lymphatic system disorders",
                "Cause skin and subcutaneous tissue disorders",
                "Cause congenital, familial, and genetic disorders",
                "Cause infections and infestations",
                "Cause respiratory and chest disorders",
                "Cause psychiatric disorders",
                "Cause renal and urinary system disorders",
                "Cause complications during pregnancy, childbirth, or perinatal period",
                "Cause ear and balance disorders",
                "Cause cardiac disorders",
                "Cause nervous system disorders",
                "Cause injury, poisoning, and procedural complications"
            ], #27
    'clintox': ['drugs approved by the FDA and passed clinical trials'] # 1 task
    }

def arg_praser() -> argparse.Namespace:
    """Parse command-line arguments for dataset processing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='+', default=[
        'esol', 'lipo', 'freesolv', 'qm9', 'bace', 'hiv',
        'bbbp', 'tox21', 'sider', 'clintox'
    ], help='list of dataset names')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = arg_praser()
    for dataset_name in args.dataset:
        logger.info(f'Processing {dataset_name}...')
        if dataset_name in regression_dataset_dict:
            task_list = regression_dataset_dict[dataset_name]
            tag = 'regression'
        elif dataset_name in classification_dataset_dict:
            task_list = classification_dataset_dict[dataset_name]
            tag = 'classification'
        else:
            logger.info(f'No task list for {dataset_name}')
            continue
        
        # Load datasets
        smiles_property_df, compare_df = load_dataset(dataset_name)
        
        # Generate QA pairs using build_qa_from_dataframes
        output_file = f'data/fgbench_qa/{dataset_name}.jsonl'
        build_qa_from_dataframes(
            smiles_property_df=smiles_property_df,
            compare_df=compare_df,
            task_list=task_list,
            tag=tag,
            output_file=output_file,
            dataset_name=dataset_name
        )
        logger.info(f'Completed processing {dataset_name}. Output saved to {output_file}')
        
