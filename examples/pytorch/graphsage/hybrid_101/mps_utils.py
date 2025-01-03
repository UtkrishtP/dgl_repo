import subprocess

# Get user id
def mps_get_user_id():
    result = subprocess.run(['id', '-u'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').rstrip()

# Start MPS daemon
def mps_daemon_start():
    result = subprocess.run(['nvidia-cuda-mps-control', '-d'], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8').rstrip())

# Start MPS server with user id
def mps_server_start(user_id):
    ps = subprocess.Popen(('echo', 'start_server -uid ' + user_id), stdout=subprocess.PIPE)
    output = subprocess.check_output(('nvidia-cuda-mps-control'), stdin=ps.stdout)
    ps.wait()

# Get created server pid
def mps_get_server_pid():
    ps = subprocess.Popen(('echo', 'get_server_list'), stdout=subprocess.PIPE)
    output = subprocess.check_output(('nvidia-cuda-mps-control'), stdin=ps.stdout)
    ps.wait()
    return output.decode('utf-8').rstrip()

# Set active thread percentage with the pid for producer
def mps_set_active_thread_percentage(server_pid, percentage):
    ps = subprocess.Popen(('echo', 'set_active_thread_percentage ' + server_pid + ' ' + str(percentage)), stdout=subprocess.PIPE)
    output = subprocess.check_output(('nvidia-cuda-mps-control'), stdin=ps.stdout)
    ps.wait()
    print('Setting set_active_thread_percentage to', output.decode('utf-8').rstrip())

# Quit MPS
def mps_quit():
    ps = subprocess.Popen(['echo', 'quit'], stdout=subprocess.PIPE)
    try:
        # Use a small timeout for demonstration (e.g., 5 seconds)
        output = subprocess.check_output(
            ['nvidia-cuda-mps-control'],
            stdin=ps.stdout,
            timeout=5  # Adjust timeout as needed
        )
    except subprocess.TimeoutExpired:
        # Instead of throwing an error, handle or ignore it here.
        output = None
        print("Timed out waiting for 'nvidia-cuda-mps-control'. Ignoring...")
    finally:
        ps.wait()

    # Optionally check the output if it succeeded
    if output is not None:
        print("MPS quit command output:", output.decode('utf-8', errors='ignore'))