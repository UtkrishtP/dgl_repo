import os, time, torch, inspect, json
import numpy as np

def setup_track_time(*args, **kwargs):
    # fix random seed
    np.random.seed(42)
    torch.random.manual_seed(42)

def setup_track_acc(*args, **kwargs):
    # fix random seed
    np.random.seed(42)
    torch.random.manual_seed(42)

def setup_track_flops(*args, **kwargs):
    # fix random seed
    np.random.seed(42)
    torch.random.manual_seed(42)

TRACK_UNITS = {
    "time": "s",
    "acc": "%",
    "flops": "GFLOPS",
}

TRACK_SETUP = {
    "time": setup_track_time,
    "acc": setup_track_acc,
    "flops": setup_track_flops,
}

class TestFilter:
    def __init__(self):
        self.conf = None
        if "DGL_REG_CONF" in os.environ:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(
                current_dir, "../../", os.environ["DGL_REG_CONF"]
            )
            with open(path, "r") as f:
                self.conf = json.load(f)
            if "INSTANCE_TYPE" in os.environ:
                instance_type = os.environ["INSTANCE_TYPE"]
            else:
                raise Exception(
                    "Must set both DGL_REG_CONF and INSTANCE_TYPE as env"
                )
            self.enabled_tests = self.conf[instance_type]["tests"]
        else:
            import logging

            # logging.warning("No regression test conf file specified")

    def check(self, func):
        funcfullname = inspect.getmodule(func).__name__ + "." + func.__name__
        if self.conf is None:
            return True
        else:
            for enabled_testname in self.enabled_tests:
                if enabled_testname in funcfullname:
                    return True
            return False

filter = TestFilter()
def benchmark(track_type, timeout=60):
    """Decorator for indicating the benchmark type.

    Parameters
    ----------
    track_type : str
        Type. Must be either:

            - 'time' : For timing. Unit: second.
            - 'acc' : For accuracy. Unit: percentage, value between 0 and 100.
            - 'flops' : Unit: GFlops, number of floating point operations per second.
    timeout : int
        Timeout threshold in second.

    Examples
    --------

    .. code::
        @benchmark('time')
        def foo():
            pass
    """
    assert track_type in ["time", "acc", "flops"]

    def _wrapper(func):
        func.unit = TRACK_UNITS[track_type]
        func.setup = TRACK_SETUP[track_type]
        func.timeout = timeout
        if not filter.check(func):
            # skip if not enabled
            func.benchmark_name = "skip_" + func.__name__
        return func

    return _wrapper

def get_bench_device():
    device = os.environ.get("DGL_BENCH_DEVICE", "cpu")
    if device.lower() == "gpu":
        return "cuda:0"
    else:
        return device

class Timer:
    def __init__(self, device=None):
        self.timer = time.perf_counter
        if device is None:
            self.device = get_bench_device()
        else:
            self.device = device

    def __enter__(self):
        if self.device == "cuda:0":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            # print("cuda timer")
        else:
            self.tic = self.timer()
        return self

    def __exit__(self, type, value, traceback):
        if self.device == "cuda:0":
            self.end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            self.elapsed_secs = (
                self.start_event.elapsed_time(self.end_event) / 1e3
            )
        else:
            self.elapsed_secs = self.timer() - self.tic
