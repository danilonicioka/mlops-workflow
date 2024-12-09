from concurrent.futures import ProcessPoolExecutor

def run_script(arg):
    import subprocess
    try:
        # Run the subprocess and capture output
        result = subprocess.run(
            ["python", "/pv/tests/concurrent-single.py"] + arg.split(),
            text=True,  # Ensure output is returned as a string
            capture_output=True,
            check=True  # Raise exception for non-zero exit code
        )
        # Return the standard output
        return f"Result for {arg}:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        # Return the error message
        return f"Error for {arg}:\n{e.stderr}"

args = ["--execn 1", "--execn 2", "--execn 3", "--execn 4"]

# Run processes concurrently
with ProcessPoolExecutor() as executor:
    results = executor.map(run_script, args)

# Print all results after execution
for output in results:
    print(output)
