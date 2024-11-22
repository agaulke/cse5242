import subprocess
import csv
import os
import re
from tqdm import tqdm 

# Path to your C executable
EXECUTABLE_PATH = "./db5242"

# Range of values for command-line arguments
N_VALUES = range(10, 10**7, 50000)  # Example values for the first argument
N_EXPONENTS = range(1, 8)
R_VALUE = 15
# BULK_BIN_PATTERN = r"bulk_bin.*?(\d+)\s+microseconds\s+or\s+([\d\.]+)\s+microseconds per search"
BULK_BIN_SEARCH_4X_PATTERN = r"bulk_bin_search_4x.*?(\d+)\s+microseconds\s+or\s+([\d\.]+)\s+microseconds per search"

# Output CSV file
CSV_FILE = "results.csv"

def run_c_program(n):
    """
    Run the C executable with given arguments and capture its output.
    """
    try:
        # Construct the command with arguments
        command = [EXECUTABLE_PATH, str(n), str(5), str(5), str(5), str(R_VALUE)]
        # Execute the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()  # Return the output
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None
    
def extract_values(output, pattern):
    """
    Extract the two numbers from the output using the regex pattern.
    """
    match = re.search(pattern, output)
    if match:
        return int(match.group(1)), float(match.group(2))  # Return as tuple (integer, float)
    return None, None


def main():
    # Ensure the executable exists
    if not os.path.isfile(EXECUTABLE_PATH):
        print(f"Executable {EXECUTABLE_PATH} not found.")
        return

    # Open the CSV file for writing
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["N","loop_microseconds", "average_microseconds_per_search"])
        #writer.writerow(["N","bulk_bin_search_4x"])

        # Loop through the values of the arguments
        for i in tqdm(N_EXPONENTS, desc="Processing", unit="iteration"):
            n = 10**i
            # Run the C program with the arguments
            output = run_c_program(n)
            if output:
                # Parse the output and write to the CSV file
                bulk_bin_total, bulk_bin_per_search = extract_values(output, BULK_BIN_SEARCH_4X_PATTERN)
                # Extract values for bulk_bin_search_4x
                #bulk_bin_search_4x_total, bulk_bin_search_4x_per_search = extract_values(output, BULK_BIN_SEARCH_4X_PATTERN)
                writer.writerow([n,bulk_bin_total, bulk_bin_per_search])
                #writer.writerow([n,bulk_bin_search_4x_time])

if __name__ == "__main__":
    main()
