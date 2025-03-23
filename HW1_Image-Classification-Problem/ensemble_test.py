"""
This script performs an ensemble voting procedure over multiple ZIP files
containing CSV files with image prediction results. The final predictions 
are determined by choosing the most common predicted label for each image 
across all CSV files and saving the results into a new CSV file.
"""
import zipfile
import os
from collections import Counter
import pandas as pd

# Folder path containing multiple ZIP files
folder_path = './local/csv_0.94/'

# Dictionary to store vote results
image_votes = {}

# Iterate over each ZIP file in the folder
for zip_filename in os.listdir(folder_path):
    if zip_filename.endswith('.zip'):
        zip_path = os.path.join(folder_path, zip_filename)

        # Open the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List files inside the ZIP file
            zip_files = zip_ref.namelist()

            # Assume the ZIP file contains only one CSV file
            for csv_file_name in zip_files:
                if csv_file_name.endswith('.csv'):   # Only read CSV files
                    with zip_ref.open(csv_file_name) as csv_file:
                        # Read the CSV file into a DataFrame
                        df = pd.read_csv(csv_file)

                        # Process voting by grouping by image name
                        for _, row in df.iterrows():
                            image_name = row['image_name']
                            pred_label = row['pred_label']

                            if image_name not in image_votes:
                                image_votes[image_name] = []
                            image_votes[image_name].append(pred_label)

if __name__ == "__main__":
     # Perform voting (choose the most common label)
    final_predictions = []
    for image_name, labels in image_votes.items():
        # Get the most common label
        most_common_label = Counter(labels).most_common(1)[0][0]
        final_predictions.append([image_name, most_common_label])

    # Write the voting results to a new CSV file
    output_file = 'prediction.csv'
    voted_df = pd.DataFrame(final_predictions, columns=['image_name', 'pred_label'])

    # Save the DataFrame to a CSV file
    voted_df.to_csv(output_file, index=False)

    print(f"Voting results have been saved to {output_file}")
