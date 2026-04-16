"""
LaTeX formula reader for processing formula TSV files.

This module provides utilities to read and parse LaTeX formula representations
from TSV files in the latex_representation_v3 directory.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm


def read_formula_file(file_path: Path) -> pd.DataFrame:
    """
    Read a single formula TSV file.
    
    Args:
        file_path: Path to the TSV file containing formulas
        
    Returns:
        DataFrame with formula data
    """
    df = pd.read_csv(file_path, sep="\t")
    return df


def extract_formulas(df: pd.DataFrame) -> List[Dict[str, any]]:
    """
    Extract formulas and their metadata from a dataframe.
    
    Args:
        df: DataFrame read from formula TSV file
        
    Returns:
        List of dictionaries containing formula and metadata
    """
    formulas = []
    
    for idx, row in df.iterrows():
        formula_entry = {
            "id": row.get("id"),
            "visual_id": row.get("visual_id"),
            "post_id": row.get("post_id"),
            "thread_id": row.get("thread_id"),
            "type": row.get("type"),
            "formula": row.get("formula"),
            "issue": row.get("issue")
        }
        formulas.append(formula_entry)
    
    return formulas


def read_single_formula_file(file_number: int, base_path: Optional[Path] = None) -> List[Dict[str, any]]:
    """
    Read formulas from a single numbered file.
    
    Args:
        file_number: File number (e.g., 1, 2, 100)
        base_path: Base path to latex_representation_v3 folder.
                   If None, uses default project structure.
    
    Returns:
        List of formula entries
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "collection" / "formula" / "latex_representation_v3"
    
    file_path = base_path / f"{file_number}.tsv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Formula file not found: {file_path}")
    
    df = read_formula_file(file_path)
    formulas = extract_formulas(df)
    
    return formulas


def read_all_formula_files(base_path: Optional[Path] = None, num_files: int = 100) -> List[Dict[str, any]]:
    """
    Read all formula files and combine them into a single list.
    
    Args:
        base_path: Base path to latex_representation_v3 folder.
                   If None, uses default project structure.
        num_files: Number of files to read (default: 100)
    
    Returns:
        Combined list of all formula entries
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "collection" / "formula" / "latex_representation_v3"
    
    all_formulas = []
    
    with tqdm(total=num_files, desc="Reading formula files", unit="file") as pbar:
        for file_num in range(1, num_files + 1):
            try:
                file_path = base_path / f"{file_num}.tsv"
                if file_path.exists():
                    df = read_formula_file(file_path)
                    formulas = extract_formulas(df)
                    all_formulas.extend(formulas)
                    pbar.update(1)
                else:
                    pbar.update(1)
            except Exception as e:
                print(f"\nError reading file {file_num}: {e}")
                pbar.update(1)
                continue
    
    return all_formulas


def save_formulas_to_tsv(formulas: List[Dict[str, any]], output_path: Path) -> None:
    """
    Save formulas to a TSV file.
    
    Args:
        formulas: List of formula entries
        output_path: Path to save the TSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(formulas)
    df.to_csv(output_path, sep="\t", index=False)
    
    print(f"\nFormulas saved to: {output_path}")


def main():
    """
    Read all formula files and save to a combined TSV file.
    """
    try:
        print("Reading all formula files...")
        formulas = read_all_formula_files()
        
        print(f"\nTotal formulas extracted: {len(formulas)}")
        
        # Save to processed collection
        output_path = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "collection" / "formulas.tsv"
        save_formulas_to_tsv(formulas, output_path)
        
        print("\nFirst 5 formulas:")
        for formula in formulas[:5]:
            print(f"  ID: {formula['id']}, Formula: {formula['formula']}")
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
