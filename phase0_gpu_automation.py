import requests
import time
import paramiko
import os
import sys
from pathlib import Path

API_KEY = "YOUR_GENESIS_API_KEY"
INSTANCE_TYPE = "3090"
REGION = "eu-west-2"
IMAGE = "ubuntu-22.04-pytorch"
USERNAME = "ubuntu"
CHECK_INTERVAL = 300
IDLE_THRESHOLD = 10
INSTANCE_NAME = "gpu_test_instance_phase0"
SSH_KEY_PATH = os.path.expanduser("~/.ssh/genesis")
MAX_WAIT_TIME = 600

def validate_config():
    if API_KEY == "YOUR_GENESIS_API_KEY" or not API_KEY.strip():
        raise ValueError("❌ Please set your Genesis Cloud API key in API_KEY.")
    if not Path(SSH_KEY_PATH).exists():
        raise FileNotFoundError(f"❌ SSH private key not found at: {SSH_KEY_PATH}")
    print("✅ Configuration validated.")

def create_instance():
    url = "https://api.genesiscloud.com/v1/instances"  # ✅ CORRECT: no spaces
    payload = {"name": INSTANCE_NAME, "region": REGION, "instance_type": INSTANCE_TYPE, "image": IMAGE}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 201:
            instance_id = response.json()["id"]
            print(f"[INFO] Instance created: {instance_id}")
            return instance_id
        else:
            print(f"[ERROR] Failed to create instance: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception during instance creation: {e}")
        return None

def wait_for_running(instance_id):
    url = f"https://api.genesiscloud.com/v1/instances/{instance_id}"  # ✅ CORRECT
    headers = {"Authorization": f"Bearer {API_KEY}"}
    start_time = time.time()
    while True:
        if time.time() - start_time > MAX_WAIT_TIME:
            print("[ERROR] Timeout waiting for instance to become running.")
            return None
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"[WARN] API error: {resp.status_code}")
                time.sleep(10)
                continue
            data = resp.json()
            status = data.get("status")
            print(f"[INFO] Instance status: {status}")
            if status == "running":
                ip_address = data.get("ip_address")
                if ip_address:
                    print(f"[INFO] Instance ready: {ip_address}")
                    return ip_address
                else:
                    print("[WARN] No IP address assigned yet.")
            elif status in ("error", "failed"):
                print(f"[ERROR] Instance failed to start: {data}")
                return None
        except Exception as e:
            print(f"[WARN] Exception while polling: {e}")
        time.sleep(10)

def run_gpu_test(ip_address):
    print("[INFO] Running GPU test via SSH...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        private_key = paramiko.RSAKey.from_private_key_file(SSH_KEY_PATH)
        ssh.connect(ip_address, username=USERNAME, pkey=private_key, timeout=30)
        stdin, stdout, stderr = ssh.exec_command("nvidia-smi")
        output = stdout.read().decode()
        error = stderr.read().decode()
        print(f"[INFO] GPU Test Output:\n{output}")
        if error:
            print(f"[WARN] Stderr: {error}")
    except Exception as e:
        print(f"[ERROR] SSH/GPU test failed: {e}")
    finally:
        ssh.close()

def monitor_and_shutdown(instance_id, ip_address):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        private_key = paramiko.RSAKey.from_private_key_file(SSH_KEY_PATH)
        ssh.connect(ip_address, username=USERNAME, pkey=private_key, timeout=30)
        while True:
            try:
                stdin, stdout, stderr = ssh.exec_command(
                    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
                )
                output = stdout.read().decode().strip()
                error = stderr.read().decode().strip()
                if error:
                    utilization = 0
                elif output:
                    util_values = [int(x) for x in output.split('\n') if x.strip().isdigit()]
                    utilization = max(util_values) if util_values else 0
                else:
                    utilization = 0
                print(f"[INFO] Current GPU utilization: {utilization}%")
                if utilization < IDLE_THRESHOLD:
                    print("[INFO] GPU idle detected. Shutting down instance...")
                    shutdown_instance(instance_id)
                    break
                time.sleep(CHECK_INTERVAL)
            except Exception as e:
                print(f"[WARN] Monitoring error: {e}")
                time.sleep(60)
    except Exception as e:
        print(f"[ERROR] Failed to connect for monitoring: {e}")
    finally:
        ssh.close()

def shutdown_instance(instance_id):
    url = f"https://api.genesiscloud.com/v1/instances/{instance_id}/shutdown"  # ✅ CORRECT
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        response = requests.post(url, headers=headers, timeout=30)
        if response.status_code == 200:
            print("[INFO] Instance shutdown successfully.")
        else:
            print(f"[ERROR] Failed to shutdown instance: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[ERROR] Exception during shutdown: {e}")

def main():
    validate_config()
    instance_id = None
    ip = None
    try:
        instance_id = create_instance()
        if not instance_id:
            sys.exit(1)
        ip = wait_for_running(instance_id)
        if not ip:
            sys.exit(1)
        run_gpu_test(ip)
        monitor_and_shutdown(instance_id, ip)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[CRITICAL] Unexpected error: {e}")
    finally:
        if instance_id and ip:
            print("[INFO] Ensuring instance is stopped...")
            shutdown_instance(instance_id)

if __name__ == "__main__":
    main()
