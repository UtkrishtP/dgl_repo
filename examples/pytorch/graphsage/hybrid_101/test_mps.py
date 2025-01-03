import torch, time, argparse
from mps_utils import *
def get_sm_count(device_index=0):
    """
    Returns the number of streaming multiprocessors (SMs)
    on the specified GPU device.
    """
    props = torch.cuda.get_device_properties(device_index)
    return props.multi_processor_count

def child(event,):
    print("Child set 20: ", get_sm_count(0))
    event.wait()
    print("Child after main set to 80: ", get_sm_count(0))
    time.sleep(1)

def launch_child_mps():
    print("Main Start: ", get_sm_count(0))
    event = torch.multiprocessing.Event()
    event.clear()
    user_id = mps_get_user_id()
    mps_daemon_start()
    mps_server_start(user_id)
    server_pid = mps_get_server_pid()
    mps_set_active_thread_percentage(server_pid, 20)
    print("Main set 20: ", get_sm_count(0))
    mp = torch.multiprocessing.get_context('spawn')
    p = mp.Process(target=child, args=(event,))
    p.start()
    time.sleep(1)
    mps_set_active_thread_percentage(server_pid, 80)
    print("Main set 80: ", get_sm_count(0))
    event.set()
    time.sleep(2)
    p.join()
    mps_quit()
    print("Main End: ", get_sm_count(0))

def start_mps(pct):
    user_id = mps_get_user_id()
    mps_daemon_start()
    mps_server_start(user_id)
    server_pid = mps_get_server_pid()
    # mps_set_active_thread_percentage(server_pid, pct)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pct",
        type=int,
        default=20,
    )
    args = parser.parse_args()
    start_mps(args.pct)