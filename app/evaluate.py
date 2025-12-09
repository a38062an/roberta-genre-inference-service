import pandas as pd
import re
import csv
import sys
import os
from sklearn.metrics import f1_score
from typing import List, Tuple, Any

def read_data(submission_file_path: str, gold_standard_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Read submission and gold standard files.
    Extract student ID from filename.
    """
    # Try to find student ID from the filename (looks for 8 digit numbers)
    id_regex = r'\d{8}'

    user_id = re.findall(id_regex, submission_file_path)
    if user_id:
        user_id = user_id[0]
    else:
        user_id = 'Unknown'

    submission_df = pd.read_csv(submission_file_path, sep=',', header=None,
                                quoting=csv.QUOTE_NONE, encoding='utf-8')

    gold_standard_df = pd.read_csv(gold_standard_file_path, header=None)

    # Remove columns 1 and 2 (keep only ID and labels)
    # The gold standard seems to have metadata in cols 1 and 2 in the notebook logic
    gold_standard_df = gold_standard_df.drop([1, 2], axis=1)
    # Skip header row
    gold_standard_df = gold_standard_df.iloc[1:]

    return submission_df, gold_standard_df, user_id


def match_and_prepare_data(submission_df: pd.DataFrame, gold_standard_df: pd.DataFrame, user_id: str) -> Tuple[List[List[int]], List[List[int]], List[str]]:
    """
    Match submission rows with gold standard rows by ID.
    Prepare data for evaluation.
    """
    gold_standard_labels = []
    submission_labels = []
    missed_rows = []
    submission_df_copy = submission_df.copy()

    # Match each gold standard row with submission
    for index, row in gold_standard_df.iterrows():
        row = row.reset_index(drop=True)
        row_found = False
        row_id = row[0]

        # Extract gold standard labels
        row_labels = [int(row[i]) for i in range(1, len(row))]
        gold_standard_labels.append(row_labels)

        # Find corresponding submission row
        for sub_index, submission_row in submission_df_copy.iterrows():
            # Robust string comparison
            if str(submission_row[0]).strip() == str(row_id).strip():
                try:
                    # Extract submission labels
                    submission_row_labels = [int(submission_row[i]) for i in range(1, len(submission_row))]
                except:
                    # Handle malformed labels (take first character if multi-digit)
                    submission_row_labels = [int(str(submission_row[i])[0]) for i in range(1, len(submission_row))]

                submission_labels.append(submission_row_labels)
                row_found = True
                submission_df_copy.drop(sub_index, inplace=True)
                break

        if not row_found:
            # If row is missing, add inverse labels (worst possible prediction)
            missed_rows.append(row_id)
            submission_labels.append([0 if label == 1 else 1 for label in row_labels])

    return gold_standard_labels, submission_labels, missed_rows


def evaluate_submission(gold_standard_labels: List[List[int]], submission_labels: List[List[int]]) -> float:
    """
    Calculate weighted F1 score.
    """
    # Calculate weighted F1 score (accounts for class imbalance)
    f1_weighted = f1_score(gold_standard_labels, submission_labels, average='weighted')
    return f1_weighted


def evaluate(submission_file: str, gold_standard_file: str) -> float:
    """
    Main function to run the submission evaluation.
    """

    # Check if files exist
    if not os.path.exists(submission_file):
        raise FileNotFoundError(f"Submission file '{submission_file}' not found!")

    if not os.path.exists(gold_standard_file):
        raise FileNotFoundError(f"Gold standard file '{gold_standard_file}' not found!")

    try:
        # Step 1: Read data
        submission_df, gold_standard_df, user_id = read_data(submission_file, gold_standard_file)

        # Step 2: Match and prepare data
        gold_standard_labels, submission_labels, missed_rows = match_and_prepare_data(
            submission_df, gold_standard_df, user_id
        )

        # Step 3: Evaluate
        f1_weighted = evaluate_submission(gold_standard_labels, submission_labels)

        # Step 4: Report results
        print(f"\nEvaluation Results for ID: {user_id}")
        print(f"Weighted F1 Score: {f1_weighted:.4f}")
        
        if missed_rows:
            print(f"WARNING: {len(missed_rows)} rows were missing from submission and penalized.")
        else:
            print("Data Completeness: 100% (All rows found)")
            
        return f1_weighted

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m app.evaluate <submission_csv> <gold_standard_csv>")
        sys.exit(1)
        
    evaluate(sys.argv[1], sys.argv[2])
