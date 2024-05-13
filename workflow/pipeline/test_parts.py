import logging
from subprocess import run

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_dvc_pull(cloned_dir):
    """Perform a DVC pull to synchronize local data with the remote repository."""
    try:
        # Run the `dvc pull` command
        result = run(['dvc', 'pull'], cwd=cloned_dir, capture_output=True, text=True)
        
        # Check if the command executed successfully
        if result.returncode != 0:
            # Log and raise an error if the command failed
            error_message = f"dvc pull failed with error: {result.stderr}"
            logger.error(error_message)
            raise Exception(error_message)
        
        # Log successful operation
        logger.info("Successfully pulled data from remote DVC repository")
        
    except Exception as e:
        # Log and handle the error
        logger.error(f"Error occurred during dvc pull: {e}")
        raise