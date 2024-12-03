from handler import MyHandler
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="Process some arguments.")

# Add arguments
parser.add_argument(
    "--execn", type=str, required=False, help="Number of the execution"
)

# Parse the arguments
args = parser.parse_args()

handler = MyHandler()

data = [{"data": [13,13,13,0,13,13,13,13,-76,-76,-81,-76,-78.5,-76,-76,-7,-7,-12,-7,-9.5,-7,-7,12,12,7,12,9.5,12,12]}]

result = handler.handle(data)

print("exec n: ", {args.execn}, "\nprediction: ", result)