# Reference Counting and Correlation Analysis

This folder contains scripts for analyzing self-references and other-references in German transcripts from client-therapist conversations.

## What does this script do?

The `count_references.py` script analyzes speech turns from transcript JSON files and counts:

1. **Self-references** (German forms):
   - `ich`, `mich`, `mir`, `meiner`, `mein`, `meine`, `meines`, `meinem`, `meinen`, `meins`

2. **Other-references** (German forms):
   - **Formal (Sie-form)**: `Sie`, `Ihnen`, `Ihrer`, `Ihr`, `Ihre`, `Ihres`, `Ihrem`, `Ihren` (must be capitalized)
   - **Informal (du-form)**: `du`, `dich`, `dir`, `deiner`, `dein`, `deine`, `deines`, `deinem`, `deinen`, `deins` (case-insensitive)

### Correlation Analysis

The script computes **four Pearson correlations**. Each correlation measures the relationship between **reference count** and **word count** within individual speech turns, separated by speaker role and reference type:

| Correlation | What is correlated |
|-------------|-------------------|
| 1. Self-refs (Client leading) | Number of self-references (ich, mich, mir...) **vs.** word count in each **client** speech turn |
| 2. Self-refs (Interviewer leading) | Number of self-references **vs.** word count in each **interviewer** speech turn |
| 3. Other-refs (Client leading) | Number of other-references (Sie, du...) **vs.** word count in each **client** speech turn |
| 4. Other-refs (Interviewer leading) | Number of other-references **vs.** word count in each **interviewer** speech turn |


**Expected patterns in therapy transcripts:**
- **High r for Client self-refs**: Clients talk about themselves (ich, mich, mir)
- **Low r for Interviewer self-refs**: Therapists rarely use self-references
- **Low r for Client other-refs**: Clients rarely address the therapist directly
- **Moderate r for Interviewer other-refs**: Therapists address the client (Sie, du)

## Setup

### Python Version

Python 3.8 or higher is recommended.

### Windows (PowerShell)

```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### macOS / Linux (bash/zsh)

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Alternative: Install without virtual environment

```bash
pip install pandas numpy scipy
```

## Usage

### Basic Command

```bash
python count_references.py --input /path/to/transcripts --output /path/to/results
```

### Arguments

- `--input`: Path to directory containing JSON transcript files
- `--output`: Path to output directory for results (CSV files and correlation summary). The script creates the result/ directory by itself.
- `--verbose`: (Optional) Enable verbose output for debugging

### Example

```bash
python count_references.py --input "C:\Users\mlut\OneDrive - ITU\Desktop\msb\results\transcripts" --output "./results"
```

## Input Format

The script expects JSON files with the following structure:

```json
[
    {
        "text": "Ja, mir geht es schon seit längerer Zeit nicht gut.",
        "start": 15320,
        "end": 24112,
        "speaker_id": "client"
    },
    {
        "text": "Und ist das neu dazugekommen?",
        "start": 73741,
        "end": 78500,
        "speaker_id": "therapist"
    }
]
```

Each entry represents a speech turn with:
- `text`: The transcribed text
- `start`: Start timestamp in milliseconds
- `end`: End timestamp in milliseconds
- `speaker_id`: Either `"client"` or `"therapist"`

## Output

The script generates:

1. **`reference_counts.csv`**: Detailed counts per speech turn
2. **`correlation_summary.txt`**: Summary of the four main correlations
3. **`aggregated_results.csv`**: Aggregated statistics per file
4. **`correlations.json`**: Correlation results in JSON format

### Plot

The script also generates:

- **`cumulative_references_plot.png`**: Shows cumulative reference counts over the session timeline for all four categories

## Important Notes on German Linguistics

- **Formal Sie-forms** must be capitalized to distinguish from third-person pronouns (sie/ihr = she/they)
- **Informal du-forms** are counted regardless of capitalization (German convention)
- The script uses word boundary matching to avoid counting partial matches within longer words
