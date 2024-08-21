import json

labels = "/home/maxime/prg/phd/dev/gonad/Export v2 project - dev_gonad - 2_28_2024.ndjson"
# Placeholder for the counts of eggs per image
egg_counts_per_image = {}

# Assuming 'annotations.njson' is your file with the annotations
with open(labels, 'r') as file:
    for line in file:
        # Parse the JSON object from the current line
        json_obj = json.loads(line)
        
        # Extract the external ID or any identifier for the image
        image_id = json_obj["data_row"]["external_id"]
        
        # Initialize count of eggs to 0 (assuming there might be no annotations)
        num_eggs = 0
        
        # Check if the annotations and objects are present
        projects = json_obj.get("projects", {})
        if projects:
            first_project_key = list(projects.keys())[0]
            labels = projects[first_project_key].get("labels", [])
            if labels:
                annotations = labels[0].get("annotations", {})
                objects = annotations.get("objects", [])
                
                # Now that we've safely accessed the objects, count the number of "eggs"
                num_eggs = len(objects)
        
        # Store the count in the dictionary
        egg_counts_per_image[image_id] = num_eggs

# Now, egg_counts_per_image will contain the counts of eggs per image, including 0 for images without annotations



import csv


# Specify the filename for the CSV file
csv_filename = 'egg_counts_per_image.csv'

# Open the file in write mode
with open(csv_filename, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)
    
    # Write the header row
    csv_writer.writerow(['image_id', 'egg_count'])
    
    # Iterate over the dictionary and write the image ID and egg count to the CSV file
    for image_id, egg_count in egg_counts_per_image.items():
        csv_writer.writerow([image_id, egg_count])

# After running this, 'egg_counts_per_image.csv' will contain your data
