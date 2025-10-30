import os
import shutil
import time
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    log.info("--- DEBUG ENTRYPOINT STARTED ---")
    
    code_path = "/opt/ml/code"
    output_path = "/opt/ml/output/data/debug_snapshot"
    
    log.info(f"Source code path: {code_path}")
    log.info(f"Destination S3 path (via /opt/ml/output): {output_path}")
    
    try:
        # Ensure the destination exists
        os.makedirs(output_path, exist_ok=True)
        
        # Copy the entire /opt/ml/code directory
        shutil.copytree(code_path, os.path.join(output_path, "code"))
        
        log.info("✅ Successfully copied entire /opt/ml/code directory to output.")
        
    except Exception as e:
        log.error(f"❌ FAILED to copy code: {e}", exc_info=True)
    
    log.info("Debug script finished. Sleeping for 30 seconds to ensure logs are flushed.")
    time.sleep(30)

if __name__ == "__main__":
    main()
