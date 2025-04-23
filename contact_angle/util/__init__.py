import time
from .droplet.center_coordinates import center_coordinates, read_droplet_trajectory

def elapsed_time(time_start: float) -> str:
    """Given a start time measured by `time.time()`, this function measures the time elapsed since
and formats it nicely into a string."""
    time_taken = time.time() - time_start
    if time_taken > 3600:
        hours = int(time_taken / 3600)
        time_taken -= hours * 3600
        mins = int(time_taken / 60)
        time_taken -= mins * 60
        return f'(hh:mm:ss) {hours:02}:{mins:02}:{round(time_taken):02}'
    if time_taken > 60:
        mins = int(time_taken / 60)
        time_taken -= mins * 60
        return f'{mins}mins{round(time_taken):02}s'
    return f'{time_taken:.3f}s'
