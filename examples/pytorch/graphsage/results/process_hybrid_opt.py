import math
def parse_cgg(file_path):
    with open(file_path, "r") as f:
        for line in f:
            # Split on comma and strip whitespace
            parts = [p.strip() for p in line.split(",") if p.strip()]
            # The last 4 numeric columns
            last_four = parts[-4:]  
            
            # Convert them to floats
            numbers = [float(x) for x in last_four]
            
            # Compute the average
            average = sum(numbers) / len(numbers)
            
            print(f"{average:.4f}")


def parse_table_and_compute_metrics(file_path):
    in_table = False

    count_ggg = 0
    count_gg = 0
    gpu_deque = 0.0
    set_number = 0

    with open(file_path, "r") as f:
        # We'll look for lines that define the table of interest
        for line in f:
            line_stripped = line.strip()

            # 1. Detect table start
            #    We assume the table always starts with this pattern:
            #    +----+-----------------+-----------+--------------+
            if "Variant" in line_stripped:
                in_table = True
                count_ggg = 0
                count_gg = 0
                gpu_deque = 0.0
                continue

            if not in_table:
                # Skip lines outside the table
                continue

            # 2. Detect the "End-End(s)" line
            if "End-End(s)" in line:
                # The next line should have the numeric totals: 
                # Something like:
                # |      59.4208 |        51.0089 |        8.4111 |              0 |
                _ = next(f, "")
                totals_line = next(f, "").strip()
                parts = totals_line.split("|")
                # Clean up
                parts = [p.strip() for p in parts if p.strip()]

                # Example after split & strip:
                # parts = ["59.4208", "51.0089", "8.4111", "0"]
                # print(parts)
                sum_ggg = 0.0
                sum_gg = 0.0
                sum_e2e = 0.0
                if len(parts) >= 3:
                    try:
                        sum_e2e = float(parts[0])  # E2E (s)
                        sum_ggg = float(parts[1])  # GGG Times(s)
                        sum_gg = float(parts[2])   # GG Times(s)
                    except ValueError:
                        pass

                # Compute averages based on counts
                avg_ggg = sum_ggg / count_ggg if count_ggg > 0 else 0.0
                avg_gg = sum_gg / count_gg if count_gg > 0 else 0.0
                avg_deque = gpu_deque / count_gg if count_gg > 0 else 0.0
                avg_e2e = sum_e2e / (count_ggg + count_gg) if (count_ggg + count_gg) > 0 else 0.0

                # Compute ratio
                gcd_val = math.gcd(count_ggg, count_gg)
                ratio_str = "0:0"
                if gcd_val != 0:
                    ratio_str = f"{count_ggg // gcd_val}:{count_gg // gcd_val}"

                set_number += 1
                # print(f"Set {set_number}:")
                # print(f"  Count GGG = {count_ggg}, sum GGG = {sum_ggg}, Avg GGG = {avg_ggg:.4f}")
                # print(f"  Count GG  = {count_gg},  sum GG  = {sum_gg},  Avg GG  = {avg_gg:.4f}")
                # print(f"  GGG:GG ratio = {ratio_str}\n")
                # print(f"{avg_ggg:.4f},{avg_gg:.4f},{avg_deque:.4f}")
                # print(ratio_str)
                print(f"{avg_e2e:.4f}")
                # Done with this table. Reset or keep going for next table
                in_table = False
                continue

            # 3. If we're still inside the table and haven't reached End-End(s), 
            #    parse lines for the Variant column
            if line_stripped.startswith("|"):
                parts = line.split("|")
                parts = [p.strip() for p in parts if p.strip()]
                # Example: ["0", "1735547576.5514", "GGG", "0.0000", ... ]
                if len(parts) >= 3:
                    variant = parts[2]
                    if variant == "GGG":
                        count_ggg += 1
                    elif variant == "GG":
                        count_gg += 1
                        gpu_deque += float(parts[5])
        # print(f"Total sets: {set_number}")

keys_of_interest = {
        "diff  (#MBs)": [],
        "slack (#MBs)": [],
        "CPU Shared read": [],
        "Enqueue": [],
        "Transfer": [],
        "Sampling time": [],
        "# mini_batch": [],
        "mfg_transfer (s)": [],
        "MFG stalls ET": [],
        "ET Stall(s)": [],
        "t_sample (s)": [],
        "Wait time": [],
    }

def parse_additional_metrics(file_path):
    """
    Parse the file and extract the following metrics:
      1. diff  (#MBs)
      2. slack (#MBs)
      3. CPU Shared read
      4. Enqueue
      5. Transfer
      6. Sampling time

    Returns a dictionary mapping each metric to a list of its float values, for example:
    {
      "diff  (#MBs)": [125.0, 130.0],
      "slack (#MBs)": [102.0, 110.0],
      "CPU Shared read": [0.1104],
      "Enqueue": [0.6812],
      "Transfer": [0.5696],
      "Sampling time": [22.3504]
    }
    """
    # Initialize dictionary with empty lists to store multiple values for each metric
    results = {key: [] for key in keys_of_interest}

    with open(file_path, "r") as f:
        for line in f:
            # Only parse lines that start with "|", which indicate relevant data
            if line.strip().startswith("|"):
                # Split the line by pipe characters and strip extra spaces
                parts = [p.strip() for p in line.split("|") if p.strip()]
                
                # We expect the line to have at least 3 columns: index, metric, value
                if len(parts) >= 3:
                    metric = parts[1]
                    value_str = parts[2]
                    
                    # Check if this metric is in our list of interest
                    if metric in results:
                        try:
                            # Append the value as a float to the list for that metric
                            results[metric].append(float(value_str))
                        except ValueError:
                            # If value_str isn't a valid float, skip it
                            pass
            elif line.strip().startswith("ET Stall(s)"):
                parts = line.split(":")
                if len(parts) >= 2:
                    value_str = parts[1].strip()
                    try:
                        results["ET Stall(s)"].append(float(value_str))
                    except ValueError:
                        pass

    return results

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', help='Path to the file to process')
# Use keys_of_interest to get the list of metrics to extract
parser.add_argument('--metric', help='Metric to extract')
args = parser.parse_args()
parse_cgg(args.file_path)
# parse_table_and_compute_metrics(args.file_path)
# result = parse_additional_metrics(args.file_path)

# print(result)
# for k, v in result.items():
#     if k.startswith(args.metric):
#         for v_ in v:
#             print(f"{v_:.4f}")