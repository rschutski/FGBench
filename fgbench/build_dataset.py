from tqdm.auto import tqdm
from rdkit import DataStructs
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdRascalMCES
from collections import Counter
import numpy as np
from multiprocessing import Pool, cpu_count
from func_timeout import func_timeout, FunctionTimedOut

from accfg import (AccFG, draw_mol_with_fgs, molimg, set_atom_idx,
                   img_grid, compare_mols, draw_compare_mols,
                   draw_RascalMCES, print_fg_tree)
afg_lite = AccFG(print_load_info=True, lite=True)

import deepchem as dc
import argparse
import logging
from typing import Optional, Tuple, List, Generator, Any

logger = logging.getLogger(__name__)


def get_dataset(name: str) -> dc.data.Dataset:
    """Load a molecular dataset from DeepChem's MoleculeNet collection.
    
    This function provides a unified interface to load various molecular property
    prediction datasets from DeepChem. It supports datasets across multiple domains
    including quantum mechanics, physical chemistry, biophysics, and physiology.
    All datasets are loaded with GraphConv featurization and balancing transformers.
    
    Args:
        name: String identifier for the dataset. Supported values:
            - Quantum Mechanics: 'qm7', 'qm8', 'qm9'
            - Physical Chemistry (Regression): 'esol', 'lipo', 'freesolv'
            - Biophysics: 'hiv', 'bace', 'muv'
            - Physiology: 'bbbp', 'tox21', 'sider', 'clintox'
    
    Returns:
        dc.data.Dataset: The first dataset from the loaded datasets tuple,
            containing molecular SMILES strings and property labels.
    
    Raises:
        ValueError: If the dataset name is not recognized or not supported.
    """
    # Quantum Mechanics
    if name == 'qm7':
        tasks, datasets, transformers = dc.molnet.load_qm7(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name == 'qm8':
        tasks, datasets, transformers = dc.molnet.load_qm8(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name == 'qm9':
        tasks, datasets, transformers = dc.molnet.load_qm9(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    # Physical Chemistry (Regression)
    elif name == 'esol':
        tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name == 'lipo':
        tasks, datasets, transformers = dc.molnet.load_lipo(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name == 'freesolv':
        tasks, datasets, transformers = dc.molnet.load_freesolv(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    # Biophysics
    elif name == 'hiv':
        tasks, datasets, transformers = dc.molnet.load_hiv(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name == 'bace':
        tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name =='muv': #17 tasks
        tasks, datasets, transformers = dc.molnet.load_muv(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    # Physiology
    elif name == 'bbbp': #1 task
        tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name == 'tox21': #12 tasks
        tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    #elif name == 'toxcast': #600 tasks
    #    tasks, datasets, transformers = dc.molnet.load_toxcast(featurizer='GraphConv',splitter=None)
    elif name == 'sider': #27 tasks
        tasks, datasets, transformers = dc.molnet.load_sider(featurizer='GraphConv',splitter=None, transformers=['balancing'])    
    elif name == 'clintox': #2 tasks
        tasks, datasets, transformers = dc.molnet.load_clintox(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    return datasets[0]


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Returns a canonicalized SMILES string, or None if the input is invalid, contains
    multiple components (separated by '.'), or cannot be parsed.
    """
    try:
        if '.' in smiles: # skip multi-component SMILES
            return None
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except:
        return None


def build_smiles_property_df(dataset: dc.data.Dataset) -> pd.DataFrame:
    """
    Extracts SMILES strings and property labels from a DeepChem
    dataset and converts them into a pandas DataFrame. It canonicalizes all
    SMILES strings, removes invalid entries, and eliminates duplicate molecules
    based on SMILES strings. Property labels are stored in columns named '0',
    '1', etc., corresponding to each task in the dataset.
    
    Args:
        dataset: dc.data.Dataset object containing molecular data with 'ids'
            (SMILES strings) and 'y' (property labels) attributes.
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'smiles': Canonical SMILES strings (unique)
            - '0', '1', ...: Property labels for each task, where column names
                correspond to task indices
    
    Note:
        Invalid SMILES (None after canonicalization) and duplicate molecules
        are removed from the output DataFrame.
    """
    smiles_property_df = pd.DataFrame(dataset.ids, columns=['smiles'])
    for i in range(dataset.y.shape[1]):
        smiles_property_df[i] = dataset.y[:, i]
    smiles_property_df['smiles'] = smiles_property_df['smiles'].apply(canonicalize_smiles)
    smiles_property_df = smiles_property_df.dropna()
    smiles_property_df_dedupl = smiles_property_df.drop_duplicates(subset=['smiles'])
    logger.debug(f'removed {len(smiles_property_df) - len(smiles_property_df_dedupl)} duplicates, and get {len(smiles_property_df_dedupl)} unique smiles')
    return smiles_property_df_dedupl


def build_smiles_property_df_from_csv(csv_path: str) -> pd.DataFrame:
    """Loads molecular data from a CSV file, canonicalizes SMILES
    strings, removes invalid entries, and eliminates duplicate molecules.
    The input CSV must contain 'smiles' and 'y' columns, where 'y' represents
    the property label.
    
    Args:
        csv_path: String path to the CSV file containing molecular data.
            Must contain columns 'smiles' and 'y'.
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'smiles': Canonical SMILES strings (unique)
            - 'y': Property labels
    
    Raises:
        AssertionError: If the CSV file does not contain required columns
            'smiles' and 'y'.
    
    Note:
        Invalid SMILES (None after canonicalization) and duplicate molecules
        are removed from the output DataFrame.
    """
    # Load the CSV file into a DataFrame, ensuring the 'smiles' and 'y' columns are present
    df = pd.read_csv(csv_path)
    assert 'smiles' in df.columns
    assert 'y' in df.columns
    smiles_property_df = df[['smiles', 'y']].copy()
    smiles_property_df['smiles'] = smiles_property_df['smiles'].apply(canonicalize_smiles)
    smiles_property_df = smiles_property_df.dropna()
    smiles_property_df_dedupl = smiles_property_df.drop_duplicates(subset=['smiles'])
    logger.debug(f'removed {len(smiles_property_df) - len(smiles_property_df_dedupl)} duplicates, and get {len(smiles_property_df_dedupl)} unique smiles')
    return smiles_property_df_dedupl


def get_similarity_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Tanimoto similarity matrix for molecules in a DataFrame.
    
    This function calculates the Tanimoto similarity (Jaccard coefficient) between
    all pairs of molecules using Morgan fingerprints (radius=2, size=512). The
    similarity matrix is stored in a DataFrame where rows and columns are indexed
    by SMILES strings. Only the upper triangular portion is computed for efficiency.
    
    Args:
        df: pandas DataFrame containing a 'smiles' column with molecular SMILES
            strings. All other columns are ignored.
    
    Returns:
        pd.DataFrame: Square DataFrame with SMILES strings as both index and columns.
            Each cell (i, j) contains the Tanimoto similarity between molecule i
            and molecule j. The matrix is symmetric, but only the upper triangular
            portion is explicitly computed.
    
    Raises:
        AssertionError: If the DataFrame does not contain a 'smiles' column.
    
    Note:
        This function can be memory-intensive for large datasets (O(n²) space
        complexity). For datasets with >10,000 molecules, consider using
        `compare_large_dataset` instead.
    """
    assert 'smiles' in df.columns
    fpgen = AllChem.GetMorganGenerator(radius=2,fpSize=512)
    smi_list = df['smiles'].tolist()
    mol_list = [Chem.MolFromSmiles(smi) for smi in smi_list]
    fp_list = [fpgen.GetFingerprint(mol) for mol in mol_list]  
    similarity_df = pd.DataFrame(index=smi_list, columns=smi_list)
    for i in tqdm(range(len(smi_list)-1)):
        target_smi = smi_list[i]
        s = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[i+1:])
        similarity_df.loc[target_smi, smi_list[i+1:]] = s
    return similarity_df


def compare_mols_in_df(mol_1: Any, mol_2: Any, afg: AccFG = afg_lite, similarityThreshold: float = 0.2, canonical: bool = False) -> Optional[Tuple[List[Any], List[Any]]]:
    """
    Safely compares two molecules and identifies their functional group differences using AccFG.
    Finds the maximum common edge substructure (MCES) between the two molecules,
    computed by RDKit Rascal MCES, and then enforced as the identical remainder after
    removing functional-group differences.
    
    Returns a pair of “difference descriptors”, one for the target molecule and one for the reference:
    ((unique_target_fgs_atoms, target_remain_alkane),
    (unique_ref_fgs_atoms,   ref_remain_alkane))
    1) unique_target_fgs_atoms / unique_ref_fgs_atoms:
        Each entry is (fg_name, count, atoms_list):
            - fg_name: name of the functional group (e.g. "Alkene", "Ether").
            - count: how many such FGs are unique to that molecule vs the other.
            - atoms_list: list of atom index lists, each sub‑list giving the atom indices (via atomNote) that form one instance of that functional group.
    2) target_remain_alkane / ref_remain_alkane:
        Each entry is (alkane_name, count, atoms_list):
            - alkane_name: e.g. "C1 alkane", "C2 alkane".
            - count: number of fragments of this alkane type.
            - atoms_list: list of atom index lists (again in terms of annotated indices), one per alkane fragment.

    Returns None if the comparison fails.    
    """
    try:
        fg_alkane_diff = compare_mols(mol_1, mol_2, afg, similarityThreshold, canonical)
        return fg_alkane_diff
    except:
        return None


def get_compare_df(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Identifies similar molecule pairs with functional group differences.
    
    This function identifies pairs of molecules with Tanimoto similarity above
    a threshold, compares them to find functional group differences, and returns
    a DataFrame containing the pairs and their differences. It filters out pairs
    with no functional group differences or failed comparisons.
    
    Args:
        df: pandas DataFrame containing a 'smiles' column with molecular SMILES
            strings.
        threshold: Float threshold for Tanimoto similarity (default: 0.7). Only
            pairs with similarity above this threshold are included.
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'target_smiles': SMILES string of the first molecule in the pair
            - 'ref_smiles': SMILES string of the second molecule in the pair
            - 'target_diff': List of functional groups in target but not in ref
            - 'ref_diff': List of functional groups in ref but not in target
        Returns an empty DataFrame with these columns if no similar pairs are found.
    
    Note:
        This function computes a full similarity matrix, which can be memory-intensive
        for large datasets. For datasets with >10,000 molecules, consider using
        `compare_large_dataset` instead.
    """
    similarity_df = get_similarity_df(df)
    condition = similarity_df > threshold
    row_indices, col_indices = np.where(condition)
    pairs = [(similarity_df.index[row], similarity_df.columns[col]) for row, col in zip(row_indices, col_indices)]

    compare_df = pd.DataFrame({'smiles_pair': pairs})
    if compare_df.empty:
        logger.info(f'Found 0 pairs of molecules with similarity > {threshold}')
        return pd.DataFrame(columns=['target_smiles','ref_smiles','target_diff','ref_diff'])
    compare_df['fg_alkane_diff'] = compare_df['smiles_pair'].apply(lambda x: compare_mols_in_df(x[0], x[1], afg_lite))
    compare_df = compare_df.dropna(subset=['fg_alkane_diff'])
    compare_df = compare_df[compare_df['fg_alkane_diff'].apply(lambda v: v != (([],[]),([],[])))]
    compare_df['target_smiles'] = compare_df['smiles_pair'].apply(lambda x: x[0])
    compare_df['ref_smiles'] = compare_df['smiles_pair'].apply(lambda x: x[1])
    compare_df['target_diff'] = compare_df['fg_alkane_diff'].apply(lambda x: x[0])
    compare_df['ref_diff'] = compare_df['fg_alkane_diff'].apply(lambda x: x[1])
    compare_df.drop(columns=['smiles_pair','fg_alkane_diff'], inplace=True)
    logger.info(f'Found {len(compare_df)} pairs of molecules with similarity > {threshold}')
    return compare_df


def generate_fingerprints(df: pd.DataFrame) -> Tuple[List[str], List[Any]]:
    """Generate Morgan fingerprints for all molecules in a DataFrame.
    
    This function computes Morgan (circular) fingerprints for all molecules
    in the input DataFrame.
    
    Args:
        df: pandas DataFrame containing a 'smiles' column with molecular SMILES
            strings.
    
    Returns:
        tuple: A tuple containing:
            - smi_list: List of SMILES strings in the same order as the DataFrame
            - fp_list: List of RDKit fingerprint objects corresponding to each SMILES
    
    Raises:
        AssertionError: If the DataFrame does not contain a 'smiles' column.
    """
    assert 'smiles' in df.columns
    fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=512)
    smi_list = df['smiles'].tolist()
    mol_list = [Chem.MolFromSmiles(smi) for smi in smi_list]
    fp_list = [fpgen.GetFingerprint(mol) for mol in mol_list]
    return smi_list, fp_list


def get_similar_pairs(smiles_list: List[str], fp_list: List[Any], threshold: float = 0.7) -> Generator[Tuple[str, str], None, None]:
    """Efficiently finds pairs of similar molecules.
    
    This function efficiently finds pairs of molecules with Tanimoto similarity.
    Yields pairs as they are found.

    Args:
        smiles_list: List of SMILES strings corresponding to the fingerprints.
        fp_list: List of RDKit fingerprint objects, must be in the same order
            as smiles_list and have the same length.
        threshold: Float threshold for Tanimoto similarity (default: 0.7). Only
            pairs with similarity above this threshold are yielded.
    
    Yields:
        tuple: Tuples of (smiles1, smiles2) representing pairs of similar molecules,
            where smiles1 comes before smiles2 in the input list.
    
    Note:
        This function only compares each molecule with molecules that come after
        it in the list, avoiding duplicate pairs and self-comparisons.
    """
    n = len(smiles_list)
    for i in tqdm(range(n - 1)):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[i + 1:])
        for j, sim in enumerate(sims):
            if sim > threshold:
                yield (smiles_list[i], smiles_list[i + 1 + j])


def compare_large_dataset(df: pd.DataFrame, threshold: float = 0.7, name: Optional[str] = None) -> None:
    """Process large datasets to find similar molecule pairs and their functional group differences.
    
    Generates similar pairs using fingerprints, then compares them to identify functional group
    differences. Results are written incrementally to CSV files.
    
    The function generates two output files:
    1. A CSV file with all similar pairs (before functional group comparison)
    2. A CSV file with pairs that have valid functional group differences
    
    Args:
        df: pandas DataFrame containing a 'smiles' column with molecular SMILES
            strings.
        threshold: Float threshold for Tanimoto similarity (default: 0.7). Only
            pairs with similarity above this threshold are considered.
        name: String identifier for the dataset, used to name output files.
            If None, output files will not be created. Defaults to None.
    
    Returns:
        None: The function writes files to disk but does not return a value.
            Output files are saved to:
            - 'data/molnet/{name}_similar_pairs.csv': All similar pairs
            - 'data/molnet/{name}_compare.csv': Pairs with functional group differences
    
    Note:
        This function is designed for datasets with >10,000 molecules where
        building a full similarity matrix would be memory-intensive. For smaller
        datasets, `get_compare_df` may be more efficient.
    """
    smi_list, fp_list = generate_fingerprints(df)
    logger.info("Finding similar molecule pairs...")
    similar_pairs = list(get_similar_pairs(smi_list, fp_list, threshold))
    logger.info(f'Found {len(similar_pairs)} pairs of molecules with similarity > {threshold}')
    valid_count = 0
    similar_pairs_df = pd.DataFrame(similar_pairs, columns=["target_smiles", "ref_smiles"])
    similar_pairs_df.to_csv(f"data/molnet/{name}_similar_pairs.csv", index=False)
    logger.info("Saved similar pairs to similar_pairs.csv")
    
    output_csv_path = f"data/molnet/{name}_compare.csv"
    with open(output_csv_path, 'w') as f:
        f.write("target_smiles,ref_smiles,target_diff,ref_diff\n")
        pbar = tqdm(similar_pairs, desc="Comparing molecules")
        for smi1, smi2 in pbar:
            fg_alkane_diff = compare_mols_in_df(smi1, smi2)
            if fg_alkane_diff and fg_alkane_diff != (([], []), ([], [])):
                valid_count += 1
                target_diff, ref_diff = fg_alkane_diff
                f.write(f'"{smi1}","{smi2}","{target_diff}","{ref_diff}"\n')
            pbar.set_postfix(valid=valid_count)
    logger.info(f"Similarity comparison complete. Results saved to: {output_csv_path}")
    return None

  
def run(dataset_name: str, threshold: float = 0.7) -> None:
    """Processes a dataset and generates comparison files.
    
    Loads a dataset from DeepChem, builds a DataFrame with canonicalized SMILES and property labels,
    finds similar molecule pairs, and identifies their functional group differences.
    Automatically selects the appropriate comparison method based on dataset size.
    
    Args:
        dataset_name: String identifier for the dataset (e.g., 'esol', 'hiv').
            Must be a dataset name supported by `get_dataset`.
        threshold: Float threshold for Tanimoto similarity (default: 0.7). Only
            pairs with similarity above this threshold are included in comparisons.
    
    Returns:
        None: The function writes files to disk but does not return a value.
            Output files are saved to:
            - 'data/molnet/{dataset_name}.csv': DataFrame with SMILES and properties
            - 'data/molnet/{dataset_name}_compare.csv': Comparison pairs with differences
            - 'data/molnet/{dataset_name}_similar_pairs.csv': (for large datasets only)
                All similar pairs before functional group comparison
    
    Raises:
        ValueError: If the dataset name is not recognized by `get_dataset`.
    """
    dataset = get_dataset(dataset_name)
    logger.info(f'Building smiles property dataframe for {dataset_name}...')
    smiles_property_df = build_smiles_property_df(dataset)
    smiles_property_df.to_csv(f'data/molnet/{dataset_name}.csv', index=False)
    
    if len(smiles_property_df) < 10000:
        logger.info(f'Building comparison dataframe for {dataset_name}...')
        compare_df = get_compare_df(smiles_property_df, threshold)
        compare_df.to_csv(f'data/molnet/{dataset_name}_compare.csv', index=False)
    else:
        logger.info(f'Building large comparison dataframe for {dataset_name}...')
        compare_large_dataset(smiles_property_df, threshold, name=dataset_name)
    
    return None

def arg_parser() -> argparse.Namespace:
    """Parse command-line arguments for dataset processing.
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', nargs='+', default=[
        'esol', 'lipo', 'freesolv', 'hiv', 'bace', 'muv', 
        'bbbp', 'tox21', 'sider', 'clintox'
    ], help='list of dataset names')
    parser.add_argument('--threshold',default=0.7, type=float, help='threshold')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = arg_parser()
    for dataset_name in args.dataset:
        logger.info(f'Processing {dataset_name}...')
        run(dataset_name, args.threshold)






