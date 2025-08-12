import argparse
import sys
import os
from bhost_parser import get_hosts_info

def mem_str_to_mb(mem_str):
    if not mem_str:
        return 0
    mem_str = mem_str.strip().upper()
    if mem_str.endswith('M'):
        return int(float(mem_str[:-1]))
    elif mem_str.endswith('G'):
        return int(float(mem_str[:-1]) * 1024)
    elif mem_str.endswith('T'):
        return int(float(mem_str[:-1]) * 1024 * 1024)
    else:
        try:
            return int(mem_str)
        except Exception:
            return 0

def build_feature_vector(hosts):
    vector = []
    for host in sorted(hosts.keys()):
        info = hosts[host]
        # Percentage of available cores
        max_cores = info.get('max_cores', 0)
        running_cores = info.get('running_cores', 0)
        if max_cores and max_cores > 0:
            core_pct = (max_cores - running_cores) / max_cores
        else:
            core_pct = 0.0
        # Percentage of available memory
        mem_total = mem_str_to_mb(info.get('total_mem', ''))
        mem_reserved = mem_str_to_mb(info.get('reserved_mem', ''))
        mem_max = mem_total + mem_reserved
        if mem_max > 0:
            mem_pct = mem_total / mem_max
        else:
            mem_pct = 0.0
        vector.extend([round(core_pct, 4), round(mem_pct, 4)])
    return vector

def main():
    parser = argparse.ArgumentParser(description='AutoHost LSF job submitter (simulator mode only)')
    parser.add_argument('-n', type=int, default=1, help='Number of cores')
    parser.add_argument('-mem', type=int, default=1024, help='Memory in MB')
    parser.add_argument('-t', type=int, default=60, help='Simulation running time in seconds')
    parser.add_argument('--debug', action='store_true', default=False, help='Print the generated bsub command instead of submitting (default: submit)')
    parser.add_argument('cmd', nargs=argparse.REMAINDER, help='Command to run (ignored in simulator mode)')
    args = parser.parse_args()

    hosts = get_hosts_info()
    feature_vector = build_feature_vector(hosts)
    # Append job info (num_cores, mem) to the vector
    # Discussion needed, do we know the maximum number of cores and memory per job?
    feature_vector.append(args.n/4)
    feature_vector.append(args.mem/2048)
    feature_str = ','.join(str(x) for x in feature_vector)
    auto_host_str = f"AUTO_HOST[[{feature_str}]]"

    sim_str = f"runtime={args.t} cputime={max(1, args.t-10)}"
    bsub_cmd = [
        'bsub',
        f'-n {args.n}',
        f'-R "rusage[mem={args.mem}]"',
        f'-sim "{sim_str}"',
        f'-ext "{auto_host_str}"'
    ]

    if args.cmd and len(args.cmd) > 0:
        bsub_cmd.extend(args.cmd)
    else:
        bsub_cmd.append(f'sleep {args.t}')
    cmd_str = ' '.join(bsub_cmd)
    if args.debug:
        print('Generated bsub command:')
        print(cmd_str)
    else:
        os.system(cmd_str)
    return cmd_str

if __name__ == "__main__":
    main()
