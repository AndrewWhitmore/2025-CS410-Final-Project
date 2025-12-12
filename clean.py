import csv
import re

# Input and output file paths
input_file = "data/input.csv"
output_file = "cleaned_data/freelaw.csv"

# Regex pattern: remove leading characters until we hit [a-zA-Z[(]
pattern = re.compile(r'^[^a-zA-Z\[\(]+')

# Max rows allowed in output
MAX_ROWS = 200_000
row_count = 0

with open(input_file, "r", newline="", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    text_index = header.index("text")

    writer.writerow(["text"])

    for row in reader:
        if row_count >= MAX_ROWS:
            break  # Stop writing after hitting limit

        text_value = row[text_index]

        cleaned_text = pattern.sub("", text_value)

        writer.writerow([cleaned_text])
        row_count += 1

