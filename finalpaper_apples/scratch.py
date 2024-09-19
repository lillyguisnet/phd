import os
import re

###Split the paperpile references into individual files

# Create the Paperpile folder if it doesn't exist
if not os.path.exists("/home/maxime/prg/phd/finalpaper_apples/paperpile"):
    os.makedirs("/home/maxime/prg/phd/finalpaper_apples/paperpile")

# Read the input BibTeX file
with open("/home/maxime/prg/phd/finalpaper_apples/WithabstractPaperpileReferencesSep2.bib", "r", encoding="utf-8") as f:
    content = f.read()

# Split the content into individual BibTeX entries
entries = re.split(r'\n@', content)

# Process each entry
for entry in entries:
    if entry.strip():
        # Add back the @ that was removed during splitting
        entry = '@' + entry.strip()
        
        # Extract the citation key
        match = re.search(r'@\w+{([^,]+),', entry)
        if match:
            key = match.group(1)
            
            # Create a filename based on the citation key
            filename = f"/home/maxime/prg/phd/finalpaper_apples/paperpile/{key}.bib"
            
            # Write the entry to a new file
            with open(filename, "w", encoding="utf-8") as f:
                f.write(entry)
                print(f"Created {filename}")

print("Splitting complete. Check the 'Paperpile' folder for individual BibTeX files.")




###Count the number of journals

import re
from collections import Counter


def count_journals(bib_file_path):
    journal_pattern = re.compile(r'journal\s*=\s*["{](.+?)["}]', re.IGNORECASE)
    journals = []

    with open(bib_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        entries = re.split(r'@\w+{', content)
        
        for entry in entries:
            match = journal_pattern.search(entry)
            if match:
                journals.append(match.group(1).strip())

    journal_counts = Counter(journals)
    return journal_counts

# Usage
bib_file_path = '/home/maxime/prg/phd/finalpaper_apples/NoabstractPaperpileReferencesSep1.bib'
journal_counts = count_journals(bib_file_path)

#Order the journals by number of occurences
journal_counts = sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)

#Print the results
print("Journal counts:")
for journal, count in journal_counts:
    print(f"{journal}: {count}")

#Print total number of unique journals
print(f"\nTotal number of unique journals: {len(journal_counts)}")