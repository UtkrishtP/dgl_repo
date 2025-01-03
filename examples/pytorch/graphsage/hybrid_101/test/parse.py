import torch

# Function to parse the file and extract the desired data
def parse_file_to_dict(file_path):
    indices = []
    values = []
    tensor = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # Split the line into parts
            if len(parts) < 4:  # Ensure there are enough parts to extract
                continue
            index = int(parts[1])  # Extract the first column (index)
            value = int(parts[-1])  # Extract the last column (value)
            tensor[index] = value
    
    # Convert to tensor
    return tensor

def compare_dict_values(dict2, dict1):
    """
    Check if the values at the same keys in two dictionaries are the same.
    
    :param dict1: First dictionary.
    :param dict2: Second dictionary.
    :return: A dictionary with keys indicating if the values match or mismatch.
    """
    for key in dict1.keys():
        if key in dict2:  # Ensure the key exists in both dictionaries
            if dict1[key] != dict2[key]:
                print(f"Values at key {key} do not match: {dict1[key]} != {dict2[key]}")
                return
    
def parse_file_to_tensor(file_path):
    tensor = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # Split the line into parts
            if len(parts) < 4:  # Ensure there are enough parts to extract
                continue
            value = int(parts[-1])  # Extract the last column (value)
            tensor.append(value)
    
    # Convert to tensor
    return tensor

# Path to the input file
file_path = "igb-large-producer.txt"  # Replace with your file name
# Parse the file and get the tensor
producer_tensor = parse_file_to_tensor(file_path)
file_path = "igb-large-consumer.txt"  # Replace with your file name
consumer_tensor = parse_file_to_tensor(file_path)
mb = 58594*2
pos = 58594
producer_tensor = producer_tensor[pos : pos + 5675]
consumer_tensor = consumer_tensor[:len(producer_tensor)]
print(len(producer_tensor), len(consumer_tensor))
prod_slack_four = [consumer_tensor[i + 3] - producer_tensor[i] for i in range(len(producer_tensor) - 3)]
for it, i in enumerate(prod_slack_four):
    if i < 0:
        print(i, it)
    # elif i < 1500000:
    #     print(i, it)

prod_slack = [i for i in prod_slack_four if i > 0]
print(min(prod_slack), max(prod_slack), sum(prod_slack) / len(prod_slack))
