# src/pipeline/sagemaker_compat.py
# <<< NEW MODULE: Handles compatibility with SageMaker's argument format. >>>

import os
import sys
import logging

log = logging.getLogger(__name__)

def is_running_in_sagemaker() -> bool:
    """Checks for a SageMaker environment."""
    return "SM_TRAINING_ENV" in os.environ or os.path.exists("/opt/ml")

def adapt_to_sagemaker():
    """
    If running in SageMaker, converts arguments to Hydra format and sets
    a compatible output directory.
    """
    if not is_running_in_sagemaker():
        log.info("🏠 Running in local/non-SageMaker environment.")
        return

    log.info("🔍 Detected SageMaker environment - adapting for compatibility...")
    original_args = sys.argv[1:]
    converted_args = []
    i = 0
    config_groups = {'instrument', 'experiment'}

    while i < len(original_args):
        arg = original_args[i]
        if arg.startswith('--') and not arg.startswith('--hydra'):
            key = arg[2:]
            if i + 1 < len(original_args) and not original_args[i + 1].startswith('--'):
                value = original_args[i + 1]
                converted_args.append(f"{key}={value}")
                i += 2
            else:
                converted_args.append(arg)
                i += 1
        else:
            converted_args.append(arg)
            i += 1
    
    sys.argv = [sys.argv[0]] + converted_args
    log.info(f"🔄 Converted args for Hydra: {converted_args}")

    # Set a default, non-interpolated output directory for SageMaker
    if 'hydra.run.dir' not in ' '.join(sys.argv):
        sys.argv.append('hydra.run.dir=/opt/ml/output/data')
        log.info("📁 Added SageMaker-compatible Hydra output directory.")