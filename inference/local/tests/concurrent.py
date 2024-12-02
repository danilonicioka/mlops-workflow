from concurrent.futures import ProcessPoolExecutor

def run_script():
    import subprocess
    subprocess.run(["python", "concurrent-single.py"])

args = ["--execn 1", "--execn 2", "--execn 3"]

# Run processes concurrently
with ProcessPoolExecutor() as executor:
    executor.map(run_script, args)
