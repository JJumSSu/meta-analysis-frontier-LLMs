import argparse
import re
from typing import List, Dict

from datasets import load_from_disk


def canonicalize_datasets(dataset_names: List[str]) -> Dict[str, List[str]]:
    """
    Canonicalize dataset names by grouping variations of the same dataset.

    Args:
        dataset_names: List of dataset names to canonicalize

    Returns:
        Dictionary mapping canonical names to lists of variations
    """
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def normalize_abbreviation(text: str) -> str:
        """Normalize abbreviations by removing separators and standardizing case."""
        return re.sub(r'[-_\s]', '', text.upper())

    def extract_dataset_identifiers(name: str) -> tuple:
        """
        Extract core name, difficulty level, and version indicators from a dataset name.
        Returns (core_name, difficulty, version)
        """
        # Extract content in parentheses
        difficulty_match = re.search(r'\(([^)]*(?:Easy|Medium|Hard|Complex)[^)]*)\)', name)
        difficulty = difficulty_match.group(1) if difficulty_match else ''

        # Remove parenthetical content for core name
        core_name = re.sub(r'\([^)]*\)', '', name).strip()

        # Extract version indicators
        version_indicators = set()
        indicators = [r'[-+](?:Pro|Redux|\d+\.*\d*[vV]\d+|\d+\.*\d*)', r'\+']
        for indicator in indicators:
            if match := re.search(indicator, name):
                version_indicators.add(match.group())
                core_name = re.sub(indicator, '', core_name)

        return (core_name.strip(), difficulty, ''.join(sorted(version_indicators)))

    def extract_name_components(name: str) -> tuple:
        """
        Extract components from a dataset name.
        Returns (abbrev, full_name, has_explicit_abbrev)
        """
        # Check for explicit abbreviation in parentheses
        abbrev_match = re.search(r'\(([A-Z][A-Z0-9-]*)\)', name)
        if abbrev_match:
            abbrev = abbrev_match.group(1)
            full_name = re.sub(r'\s*\([^)]*\)', '', name).strip()
            return (abbrev, full_name, True)

        # Check if the name itself is an abbreviation
        if re.match(r'^[A-Z][A-Z0-9-]{1,11}$', name):
            return (name, '', False)

        # No abbreviation found
        return ('', name, False)

    def should_group_together(name1: str, name2: str) -> bool:
        """
        Determine if two dataset names should be grouped together.
        Combines strict abbreviation matching with version/difficulty handling.
        """
        # First check version and difficulty
        core1, diff1, ver1 = extract_dataset_identifiers(name1)
        core2, diff2, ver2 = extract_dataset_identifiers(name2)

        # If difficulties are different, they're different datasets
        if diff1 and diff2 and diff1 != diff2:
            return False

        # If versions are different, they're different datasets
        if ver1 != ver2:
            return False

        # Then check for exact matches and abbreviations
        abbrev1, full1, has_explicit1 = extract_name_components(name1)
        abbrev2, full2, has_explicit2 = extract_name_components(name2)

        # If they're identical after normalization, group them
        if normalize_text(core1) == normalize_text(core2):
            return True

        # If both have abbreviations (either explicit or standalone), they must match
        if abbrev1 and abbrev2:
            return normalize_abbreviation(abbrev1) == normalize_abbreviation(abbrev2)

        # If one has an explicit abbreviation, match it with the full name
        if has_explicit1 and normalize_text(full1) == normalize_text(name2):
            return True
        if has_explicit2 and normalize_text(full2) == normalize_text(name1):
            return True

        # Compare normalized core names as a last resort
        norm1 = normalize_abbreviation(core1)
        norm2 = normalize_abbreviation(core2)
        if norm1 and norm2 and len(min(norm1, norm2)) > 3:  # Only if names are substantial
            return norm1 == norm2

        # Handle standalone full names matching abbreviations
        if abbrev1 and normalize_abbreviation(abbrev1) == normalize_abbreviation(core2):
            return True
        if abbrev2 and normalize_abbreviation(abbrev2) == normalize_abbreviation(core1):
            return True

        return False

    # Group datasets
    result: Dict[str, List[str]] = {}
    processed = set()

    # First process entries with explicit abbreviations
    abbrev_entries = [name for name in dataset_names if '(' in name and re.search(r'\([A-Z][A-Z0-9-]*\)', name)]
    for name in abbrev_entries:
        if name in processed:
            continue

        variations = [name]
        for other_name in dataset_names:
            if other_name != name and other_name not in processed:
                if should_group_together(name, other_name):
                    variations.append(other_name)
                    processed.add(other_name)

        if variations:
            result[name] = variations
            processed.add(name)

    # Then process remaining entries
    for name in dataset_names:
        if name in processed:
            continue

        variations = [name]
        for other_name in dataset_names:
            if other_name != name and other_name not in processed:
                if should_group_together(name, other_name):
                    variations.append(other_name)
                    processed.add(other_name)

        if variations:
            # Use the longest name as canonical
            canonical = max(variations, key=len)
            result[canonical] = variations
            processed.add(name)

    return result


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path', type=str)        
    args = parser.parse_args()

    dataset = load_from_disk(args.hf_ds_path)
    
    dataset_names = list(set(instance['context_augmented_table_results_extracted']['dataset'] for instance in dataset))

    print("original dataset unique length", len(dataset_names))
    
    merged_dict = canonicalize_datasets(dataset_names)  

    adjusted_dataset_names = []
    for instance in dataset:
        m = 0
        dataset_name = instance['context_augmented_table_results_extracted']['dataset']
        for key, value in merged_dict.items():
            if dataset_name.strip() in value:
                adjusted_dataset_names.append(key)
                m += 1
        if m == 0:
            adjusted_dataset_names.append(dataset_name)
        assert m <= 1, f"Multiple keys found in the merged_dict"

    print("canonicalized dataset names unique length", len(set(adjusted_dataset_names)))

    dataset = dataset.add_column('dataset_name', adjusted_dataset_names)
    dataset.save_to_disk(args.hf_ds_output_path)
