import csv

# File names
files = ['OUTPUTS_4A.TXT', 'OUTPUTS_4B.TXT', 'OUTPUTS_4C.TXT']
output_csv = 'consolidated_data.csv'

# Read data from text files and write to CSV
with open(output_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write header
    csvwriter.writerow([
        'Number of threads', 'Time_Dynamic', 'Time_Guided', 'Time_Static'
    ])

    # Initialize a dictionary to store the data
    data = {}

    # Read each file and store the data
    for i, file in enumerate(files):
        with open(file, 'r') as f:
            lines = f.readlines()
            for j in range(0, len(lines), 2):
                num_threads = int(lines[j].strip())
                time = float(lines[j + 1].strip())

                if num_threads not in data:
                    data[num_threads] = [None, None, None]
                data[num_threads][i] = time

    # Write the data to the CSV file
    for num_threads, times in sorted(data.items()):
        csvwriter.writerow([num_threads] + times)
