from handler import MyHandler
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="Process some arguments.")

# Add arguments
parser.add_argument(
    "--execn", type=int, required=False, help="Number of the execution"
)

# Parse the arguments
args = parser.parse_args()

# Initialize the handler
handler = MyHandler()

# Define different datasets for each execution
datasets = {
    1: [{"data": [13,13,13,0,13,13,13,13,-76,-76,-81,-76,-78.5,-76,-76,-7,-7,-12,-7,-9.5,-7,-7,12,12,7,12,9.5,12,12]}],
    2: [{"data": [13,13,12,0.4714045208,13,12.5,13,13,-81,-81,-82,-81,-81.5,-81,-81,-12,-12,-11,-12,-12,-12,-11.5,7,7,2,7,4.5,7,7]}],
    3: [{"data": [6,7,7,0.4714045208,7,6.5,7,7,-107,-106,-106,-106,-106.5,-106,-106,-13,-14,-14,-14,-14,-14,-13.5,2,2,2,2,2,2,2]}],
    4: [{"data": [8,8,8,0,8,8,8,8,-108,-108,-108,-108,-108,-108,-108,-13,-13,-13,-13,-13,-13,-13,2,2,2,2,2,2,2]}]
}

# Select the data based on the `--execn` argument
data = datasets.get(args.execn, [{"data": [0] * 29}])  # Default to a fallback if execn is not in datasets

# Pass the data to the handler
result = handler.handle(data)

# Print the result
print(f"Execution number: {args.execn}\nPrediction: {result}")
