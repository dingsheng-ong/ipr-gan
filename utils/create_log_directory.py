import os


def create_log_directory(directory):
    # create if not exist
    os.makedirs(directory, exist_ok=True)
    # enumerate directories
    n = os.listdir(directory)
    n = min(set(map(str, range(1, len(n) + 2))) - set(n))
    log_directory = os.path.join(directory, str(n))
    # create the log directory
    os.makedirs(log_directory, exist_ok=False)
    for sub_dir in ['checkpoint', 'sample', 'watermark']:
        os.makedirs(os.path.join(log_directory, sub_dir), exist_ok=False)
    
    return log_directory
