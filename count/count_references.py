#!/usr/bin/env python3
"""
count_references.py

Counts self-references and other-references in German transcripts,
then computes correlations based on speaker roles (client vs. therapist/interviewer).

Self-references (German forms):
    ich, mich, mir, meiner, mein, meine, meines, meinem, meinen, meins

Other-references:
    Formal (Sie-form) - must be capitalized:
        Sie, Ihnen, Ihrer, Ihr, Ihre, Ihres, Ihrem, Ihren
    Informal (du-form) - case-insensitive:
        du, dich, dir, deiner, dein, deine, deines, deinem, deinen, deins

Correlations computed:
    1. Self-references when the client is leading
    2. Self-references when the interviewer is leading
    3. Other-references when the client is leading
    4. Other-references when the interviewer is leading
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Reference word lists
# =============================================================================

# Self-reference forms (case-insensitive matching)
SELF_REFERENCES = [
    "ich", "mich", "mir", "meiner", "mein", "meine", "meines", "meinem", "meinen", "meins"
]

# Formal other-references (Sie-form) - MUST be capitalized
FORMAL_OTHER_REFERENCES = [
    "Sie", "Ihnen", "Ihrer", "Ihr", "Ihre", "Ihres", "Ihrem", "Ihren"
]

# Informal other-references (du-form) - case-insensitive
INFORMAL_OTHER_REFERENCES = [
    "du", "dich", "dir", "deiner", "dein", "deine", "deines", "deinem", "deinen", "deins"
]


# =============================================================================
# Counting functions
# =============================================================================

def count_word_occurrences(text: str, word: str, case_sensitive: bool = False) -> int:
    """
    Count occurrences of a word in text using word boundary matching.
    
    Args:
        text: The text to search in
        word: The word to count
        case_sensitive: Whether matching should be case-sensitive
    
    Returns:
        Number of occurrences found
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    # Use word boundaries to match whole words only
    pattern = r'\b' + re.escape(word) + r'\b'
    matches = re.findall(pattern, text, flags)
    
    if case_sensitive:
        # For case-sensitive matching, filter to exact case matches
        return sum(1 for m in matches if m == word)
    else:
        return len(matches)


def count_self_references(text: str) -> int:
    """
    Count all self-reference occurrences in text.
    Self-references are matched case-insensitively.
    """
    total = 0
    for word in SELF_REFERENCES:
        total += count_word_occurrences(text, word, case_sensitive=False)
    return total


def count_other_references(text: str) -> int:
    """
    Count all other-reference occurrences in text.
    
    Formal Sie-forms must be capitalized.
    Informal du-forms are counted regardless of case.
    """
    total = 0
    
    # Count formal references (case-sensitive - must be capitalized)
    for word in FORMAL_OTHER_REFERENCES:
        total += count_word_occurrences(text, word, case_sensitive=True)
    
    # Count informal references (case-insensitive)
    for word in INFORMAL_OTHER_REFERENCES:
        total += count_word_occurrences(text, word, case_sensitive=False)
    
    return total


def get_detailed_counts(text: str) -> Dict[str, int]:
    """
    Get detailed counts for each reference word category.
    
    Returns:
        Dictionary with counts for each word
    """
    counts = {}
    
    # Self-references (case-insensitive)
    for word in SELF_REFERENCES:
        counts[f"self_{word}"] = count_word_occurrences(text, word, case_sensitive=False)
    
    # Formal other-references (case-sensitive)
    for word in FORMAL_OTHER_REFERENCES:
        counts[f"formal_{word}"] = count_word_occurrences(text, word, case_sensitive=True)
    
    # Informal other-references (case-insensitive)
    for word in INFORMAL_OTHER_REFERENCES:
        counts[f"informal_{word}"] = count_word_occurrences(text, word, case_sensitive=False)
    
    return counts


# =============================================================================
# File processing
# =============================================================================

def load_transcript(filepath: str) -> List[Dict]:
    """
    Load a transcript JSON file.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        List of speech turn dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_transcript(transcript: List[Dict]) -> pd.DataFrame:
    """
    Process a transcript and count references for each speech turn.
    
    Args:
        transcript: List of speech turn dictionaries
    
    Returns:
        DataFrame with counts per speech turn
    """
    results = []
    
    for i, turn in enumerate(transcript):
        text = turn.get('text', '')
        speaker = turn.get('speaker_id', 'unknown')
        start = turn.get('start', 0)
        end = turn.get('end', 0)
        
        # Normalize speaker labels
        if speaker.lower() in ['therapist', 'interviewer']:
            speaker_normalized = 'interviewer'
        elif speaker.lower() == 'client':
            speaker_normalized = 'client'
        else:
            speaker_normalized = speaker.lower()
        
        # Count references
        self_refs = count_self_references(text)
        other_refs = count_other_references(text)
        
        results.append({
            'turn_index': i,
            'speaker': speaker_normalized,
            'start_ms': start,
            'end_ms': end,
            'text': text,
            'self_references': self_refs,
            'other_references': other_refs,
            'word_count': len(text.split())
        })
    
    return pd.DataFrame(results)


def process_directory(input_dir: str, verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Path to directory containing JSON files
        verbose: Whether to print progress
    
    Returns:
        Tuple of (combined DataFrame, dict of per-file DataFrames)
    """
    input_path = Path(input_dir)
    json_files = list(input_path.glob('*.json'))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")
    
    if verbose:
        print(f"Found {len(json_files)} JSON file(s) to process")
    
    all_results = []
    per_file_results = {}
    
    for json_file in json_files:
        if verbose:
            print(f"Processing: {json_file.name}")
        
        try:
            transcript = load_transcript(str(json_file))
            df = process_transcript(transcript)
            df['source_file'] = json_file.name
            
            all_results.append(df)
            per_file_results[json_file.name] = df
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    if not all_results:
        raise ValueError("No files could be processed successfully")
    
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df, per_file_results


# =============================================================================
# Correlation analysis
# =============================================================================

def compute_correlations(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute the four correlations:
    1. Self-references when the client is leading
    2. Self-references when the interviewer is leading
    3. Other-references when the client is leading
    4. Other-references when the interviewer is leading
    
    "Leading" is interpreted as the speaker of each speech turn.
    Correlations are computed between the speaker's reference counts
    and the corresponding counts from the other speaker.
    
    Args:
        df: DataFrame with reference counts per speech turn
    
    Returns:
        Dictionary with correlation results
    """
    results = {}
    
    # Split by speaker
    client_turns = df[df['speaker'] == 'client'].reset_index(drop=True)
    interviewer_turns = df[df['speaker'] == 'interviewer'].reset_index(drop=True)
    
    # For correlation, we need paired data. We'll compute correlations
    # between self-references and other-references within each speaker's turns.
    
    # 1. Self-references when client is leading
    if len(client_turns) > 2:
        client_self = client_turns['self_references'].values
        # Correlate with word count or other metrics
        r, p = stats.pearsonr(client_self, client_turns['word_count'].values)
        results['self_ref_client_leading'] = {
            'correlation': r,
            'p_value': p,
            'n': len(client_turns),
            'mean_count': np.mean(client_self),
            'std_count': np.std(client_self),
            'total_count': np.sum(client_self)
        }
    else:
        results['self_ref_client_leading'] = {
            'correlation': np.nan,
            'p_value': np.nan,
            'n': len(client_turns),
            'note': 'Insufficient data points'
        }
    
    # 2. Self-references when interviewer is leading
    if len(interviewer_turns) > 2:
        interviewer_self = interviewer_turns['self_references'].values
        r, p = stats.pearsonr(interviewer_self, interviewer_turns['word_count'].values)
        results['self_ref_interviewer_leading'] = {
            'correlation': r,
            'p_value': p,
            'n': len(interviewer_turns),
            'mean_count': np.mean(interviewer_self),
            'std_count': np.std(interviewer_self),
            'total_count': np.sum(interviewer_self)
        }
    else:
        results['self_ref_interviewer_leading'] = {
            'correlation': np.nan,
            'p_value': np.nan,
            'n': len(interviewer_turns),
            'note': 'Insufficient data points'
        }
    
    # 3. Other-references when client is leading
    if len(client_turns) > 2:
        client_other = client_turns['other_references'].values
        r, p = stats.pearsonr(client_other, client_turns['word_count'].values)
        results['other_ref_client_leading'] = {
            'correlation': r,
            'p_value': p,
            'n': len(client_turns),
            'mean_count': np.mean(client_other),
            'std_count': np.std(client_other),
            'total_count': np.sum(client_other)
        }
    else:
        results['other_ref_client_leading'] = {
            'correlation': np.nan,
            'p_value': np.nan,
            'n': len(client_turns),
            'note': 'Insufficient data points'
        }
    
    # 4. Other-references when interviewer is leading
    if len(interviewer_turns) > 2:
        interviewer_other = interviewer_turns['other_references'].values
        r, p = stats.pearsonr(interviewer_other, interviewer_turns['word_count'].values)
        results['other_ref_interviewer_leading'] = {
            'correlation': r,
            'p_value': p,
            'n': len(interviewer_turns),
            'mean_count': np.mean(interviewer_other),
            'std_count': np.std(interviewer_other),
            'total_count': np.sum(interviewer_other)
        }
    else:
        results['other_ref_interviewer_leading'] = {
            'correlation': np.nan,
            'p_value': np.nan,
            'n': len(interviewer_turns),
            'note': 'Insufficient data points'
        }
    
    # Additional: Cross-speaker correlations (self vs other between speakers)
    # Correlate client's self-references with interviewer's other-references (and vice versa)
    min_len = min(len(client_turns), len(interviewer_turns))
    if min_len > 2:
        # Align turns by index (sequential pairing)
        client_self = client_turns['self_references'].values[:min_len]
        client_other = client_turns['other_references'].values[:min_len]
        interviewer_self = interviewer_turns['self_references'].values[:min_len]
        interviewer_other = interviewer_turns['other_references'].values[:min_len]
        
        # Client self vs Interviewer other
        r1, p1 = stats.pearsonr(client_self, interviewer_other)
        results['cross_client_self_vs_interviewer_other'] = {
            'correlation': r1,
            'p_value': p1,
            'n': min_len
        }
        
        # Interviewer self vs Client other
        r2, p2 = stats.pearsonr(interviewer_self, client_other)
        results['cross_interviewer_self_vs_client_other'] = {
            'correlation': r2,
            'p_value': p2,
            'n': min_len
        }
    
    return results


def compute_within_speaker_correlations(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute correlations between self-references and other-references
    within each speaker's turns.
    
    Args:
        df: DataFrame with reference counts per speech turn
    
    Returns:
        Dictionary with correlation results
    """
    results = {}
    
    for speaker in ['client', 'interviewer']:
        speaker_turns = df[df['speaker'] == speaker]
        
        if len(speaker_turns) > 2:
            self_refs = speaker_turns['self_references'].values
            other_refs = speaker_turns['other_references'].values
            
            # Check for zero variance
            if np.std(self_refs) > 0 and np.std(other_refs) > 0:
                r, p = stats.pearsonr(self_refs, other_refs)
                results[f'{speaker}_self_vs_other'] = {
                    'correlation': r,
                    'p_value': p,
                    'n': len(speaker_turns),
                    'self_mean': np.mean(self_refs),
                    'other_mean': np.mean(other_refs)
                }
            else:
                results[f'{speaker}_self_vs_other'] = {
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'n': len(speaker_turns),
                    'note': 'Zero variance in one or both variables'
                }
        else:
            results[f'{speaker}_self_vs_other'] = {
                'correlation': np.nan,
                'p_value': np.nan,
                'n': len(speaker_turns),
                'note': 'Insufficient data points'
            }
    
    return results


# =============================================================================
# Plotting functions
# =============================================================================

def plot_reference_counts_over_time(df: pd.DataFrame, output_dir: str, 
                                     verbose: bool = False):
    """
    Create plots showing cumulative reference counts over the course of the session.
    
    Generates two plots:
    1. Cumulative counts over speech turns
    2. Per-turn counts with rolling average
    
    Args:
        df: DataFrame with reference counts per speech turn
        output_dir: Path to output directory for plots
        verbose: Whether to print progress
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sort by turn index to ensure chronological order
    df_sorted = df.sort_values('turn_index').reset_index(drop=True)
    
    # Create the four count series
    # Initialize arrays for cumulative counts
    n_turns = len(df_sorted)
    
    self_client_cumsum = np.zeros(n_turns)
    self_interviewer_cumsum = np.zeros(n_turns)
    other_client_cumsum = np.zeros(n_turns)
    other_interviewer_cumsum = np.zeros(n_turns)
    
    # Also track per-turn counts for the second plot
    self_client_per_turn = np.zeros(n_turns)
    self_interviewer_per_turn = np.zeros(n_turns)
    other_client_per_turn = np.zeros(n_turns)
    other_interviewer_per_turn = np.zeros(n_turns)
    
    running_self_client = 0
    running_self_interviewer = 0
    running_other_client = 0
    running_other_interviewer = 0
    
    for i, row in df_sorted.iterrows():
        if row['speaker'] == 'client':
            running_self_client += row['self_references']
            running_other_client += row['other_references']
            self_client_per_turn[i] = row['self_references']
            other_client_per_turn[i] = row['other_references']
        else:  # interviewer
            running_self_interviewer += row['self_references']
            running_other_interviewer += row['other_references']
            self_interviewer_per_turn[i] = row['self_references']
            other_interviewer_per_turn[i] = row['other_references']
        
        self_client_cumsum[i] = running_self_client
        self_interviewer_cumsum[i] = running_self_interviewer
        other_client_cumsum[i] = running_other_client
        other_interviewer_cumsum[i] = running_other_interviewer
    
    # Convert time to minutes if available
    if 'start_ms' in df_sorted.columns:
        time_axis = df_sorted['start_ms'].values / 60000  # Convert to minutes
        time_label = 'Time (minutes)'
    else:
        time_axis = np.arange(n_turns)
        time_label = 'Speech Turn'
    
    # =========================================================================
    # Plot 1: Cumulative counts over time
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time_axis, self_client_cumsum, 
            label='Self-refs (Client)', color='#2196F3', linewidth=2)
    ax.plot(time_axis, self_interviewer_cumsum, 
            label='Self-refs (Interviewer)', color='#4CAF50', linewidth=2)
    ax.plot(time_axis, other_client_cumsum, 
            label='Other-refs (Client)', color='#FF9800', linewidth=2, linestyle='--')
    ax.plot(time_axis, other_interviewer_cumsum, 
            label='Other-refs (Interviewer)', color='#9C27B0', linewidth=2, linestyle='--')
    
    ax.set_xlabel(time_label, fontsize=12)
    ax.set_ylabel('Cumulative Count', fontsize=12)
    ax.set_title('Cumulative Reference Counts Over Session', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    cumulative_plot_file = output_path / 'cumulative_references_plot.png'
    plt.savefig(cumulative_plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Saved cumulative plot to: {cumulative_plot_file}")


# =============================================================================
# Output functions
# =============================================================================

def save_results(df: pd.DataFrame, correlations: Dict, 
                 output_dir: str, verbose: bool = False):
    """
    Save results to output directory.
    
    Args:
        df: DataFrame with all reference counts
        correlations: Dictionary with main correlation results
        output_dir: Path to output directory
        verbose: Whether to print progress
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed counts
    counts_file = output_path / 'reference_counts.csv'
    df.to_csv(counts_file, index=False, encoding='utf-8')
    if verbose:
        print(f"Saved detailed counts to: {counts_file}")
    
    # Save aggregated statistics per file
    agg_stats = df.groupby('source_file').agg({
        'self_references': ['sum', 'mean', 'std'],
        'other_references': ['sum', 'mean', 'std'],
        'word_count': ['sum', 'mean'],
        'turn_index': 'count'
    }).round(3)
    agg_stats.columns = ['_'.join(col).strip() for col in agg_stats.columns.values]
    agg_stats = agg_stats.rename(columns={'turn_index_count': 'total_turns'})
    
    agg_file = output_path / 'aggregated_results.csv'
    agg_stats.to_csv(agg_file, encoding='utf-8')
    if verbose:
        print(f"Saved aggregated results to: {agg_file}")
    
    # Save correlation summary
    summary_file = output_path / 'correlation_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("REFERENCE COUNTING AND CORRELATION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total speech turns analyzed: {len(df)}\n")
        f.write(f"Client turns: {len(df[df['speaker'] == 'client'])}\n")
        f.write(f"Interviewer turns: {len(df[df['speaker'] == 'interviewer'])}\n")
        f.write(f"Total self-references: {df['self_references'].sum()}\n")
        f.write(f"Total other-references: {df['other_references'].sum()}\n")
        f.write(f"Total words: {df['word_count'].sum()}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("MAIN CORRELATIONS (Reference counts vs. Word count per turn)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. Self-references when CLIENT is leading:\n")
        f.write("-" * 40 + "\n")
        _write_correlation_result(f, correlations.get('self_ref_client_leading', {}))
        
        f.write("\n2. Self-references when INTERVIEWER is leading:\n")
        f.write("-" * 40 + "\n")
        _write_correlation_result(f, correlations.get('self_ref_interviewer_leading', {}))
        
        f.write("\n3. Other-references when CLIENT is leading:\n")
        f.write("-" * 40 + "\n")
        _write_correlation_result(f, correlations.get('other_ref_client_leading', {}))
        
        f.write("\n4. Other-references when INTERVIEWER is leading:\n")
        f.write("-" * 40 + "\n")
        _write_correlation_result(f, correlations.get('other_ref_interviewer_leading', {}))
    
    if verbose:
        print(f"Saved correlation summary to: {summary_file}")
    
    # Also save correlations as JSON for programmatic access
    corr_json_file = output_path / 'correlations.json'
    
    # Filter to only main correlations
    main_correlations = {
        k: v for k, v in correlations.items() 
        if k in ['self_ref_client_leading', 'self_ref_interviewer_leading',
                 'other_ref_client_leading', 'other_ref_interviewer_leading']
    }
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if np.isfinite(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(corr_json_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy(main_correlations), f, indent=2)
    
    if verbose:
        print(f"Saved correlations JSON to: {corr_json_file}")


def _write_correlation_result(f, result: Dict):
    """Helper function to write correlation result to file."""
    if 'note' in result:
        f.write(f"  Note: {result['note']}\n")
    
    if 'correlation' in result:
        r = result['correlation']
        if np.isnan(r) if isinstance(r, float) else False:
            f.write("  Correlation: N/A\n")
        else:
            f.write(f"  Pearson r: {r:.4f}\n")
    
    if 'p_value' in result:
        p = result['p_value']
        if not (np.isnan(p) if isinstance(p, float) else False):
            f.write(f"  p-value: {p:.4f}")
            if p < 0.001:
                f.write(" ***\n")
            elif p < 0.01:
                f.write(" **\n")
            elif p < 0.05:
                f.write(" *\n")
            else:
                f.write("\n")
    
    if 'n' in result:
        f.write(f"  N (turns): {result['n']}\n")
    
    if 'mean_count' in result:
        f.write(f"  Mean count: {result['mean_count']:.2f}\n")
    
    if 'std_count' in result:
        f.write(f"  Std. dev: {result['std_count']:.2f}\n")
    
    if 'total_count' in result:
        f.write(f"  Total count: {result['total_count']}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Count self-references and other-references in German transcripts '
                    'and compute correlations based on speaker roles.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python count_references.py --input /path/to/transcripts --output ./results
  python count_references.py --input ./data --output ./results --verbose

Reference words:
  Self-references: ich, mich, mir, meiner, mein, meine, meines, meinem, meinen, meins
  
  Other-references (formal, must be capitalized): 
    Sie, Ihnen, Ihrer, Ihr, Ihre, Ihres, Ihrem, Ihren
  
  Other-references (informal, case-insensitive):
    du, dich, dir, deiner, dein, deine, deines, deinem, deinen, deins
        '''
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to directory containing JSON transcript files'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output directory for results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Process transcripts
    if args.verbose:
        print(f"Processing transcripts from: {args.input}")
    
    try:
        combined_df, per_file_dfs = process_directory(args.input, args.verbose)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    if args.verbose:
        print(f"\nTotal speech turns processed: {len(combined_df)}")
        print(f"Client turns: {len(combined_df[combined_df['speaker'] == 'client'])}")
        print(f"Interviewer turns: {len(combined_df[combined_df['speaker'] == 'interviewer'])}")
    
    # Compute correlations
    if args.verbose:
        print("\nComputing correlations...")
    
    correlations = compute_correlations(combined_df)
    
    # Save results
    if args.verbose:
        print(f"\nSaving results to: {args.output}")
    
    save_results(combined_df, correlations, args.output, args.verbose)
    
    # Generate plots
    if args.verbose:
        print("\nGenerating plots...")
    
    plot_reference_counts_over_time(combined_df, args.output, args.verbose)
    
    # Print summary to console
    print("\n" + "=" * 50)
    print("CORRELATION SUMMARY")
    print("=" * 50)
    
    for name, key in [
        ("1. Self-references (Client leading)", "self_ref_client_leading"),
        ("2. Self-references (Interviewer leading)", "self_ref_interviewer_leading"),
        ("3. Other-references (Client leading)", "other_ref_client_leading"),
        ("4. Other-references (Interviewer leading)", "other_ref_interviewer_leading")
    ]:
        result = correlations.get(key, {})
        r = result.get('correlation', np.nan)
        p = result.get('p_value', np.nan)
        n = result.get('n', 0)
        
        r_str = f"{r:.3f}" if not (np.isnan(r) if isinstance(r, float) else False) else "N/A"
        p_str = f"{p:.4f}" if not (np.isnan(p) if isinstance(p, float) else False) else "N/A"
        
        print(f"{name}:")
        print(f"  r = {r_str}, p = {p_str}, n = {n}")
    
    print("\nResults saved to:", args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())
