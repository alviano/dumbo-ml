import importlib
import logging
import random
import time

try:
    tf = importlib.import_module("tensorflow")  # import tensorflow this way to avoid a broken dependency
except ModuleNotFoundError:
    exit("Module 'tensorflow' is required but not installed")

log = logging.getLogger(__name__)


def run_now_in_any_gpu(fun, smi):
    smi.smi_initialize()
    devices = [device for device in range(smi.smi_get_device_count())]
    random.shuffle(devices)
    log.info(f"Try with order {devices}")

    delay = 1.0
    done = False
    while not done:
        try:
            for device in devices:
                utilization = smi.smi_get_device_utilization(device)
                memory = 100 * smi.smi_get_device_memory_used(device) // smi.smi_get_device_memory_total(device)
                usage = f"(utilization {utilization}%%, memory {memory}%%)"
                if utilization > 10 or memory > 10:
                    log.info(f"Skip GPU {device} {usage}")
                    continue
                with tf.Graph().as_default():
                    with tf.device(f"/GPU:{device}"):
                        log.info(f"Try with GPU {device} {usage}")
                        result = fun()
                        done = True
        except Exception as e:
            log.info(f"Oops!", e)
        if not done:
            log.info(f"Retry in {delay:.1f} seconds...")
            time.sleep(delay)
            delay = min(300, delay * 2)

    smi.smi_shutdown()
    return result


def run_now_in_any_amd_gpu(fun):
    try:
        return run_now_in_any_gpu(fun, importlib.import_module("pyrsmi.rocml", package="pyrsmi"))
    except ModuleNotFoundError:
        exit("Module 'pyrsmi' is required but not installed")
