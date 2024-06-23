from settings import VERBOSE_MODE


def print_runtime(start_time, end_time, process_name):
    print_debug_message(
        f"Process {process_name} took {(end_time - start_time).seconds // 60} minutes"
        f"\t(Ended at {end_time.strftime('%Y/%m/%d - %H:%M:%S')})"
    )


def print_debug_message(message):
    print(message) if VERBOSE_MODE else None
