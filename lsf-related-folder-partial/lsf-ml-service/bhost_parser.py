import subprocess
import re
from typing import Dict, List, Optional


class HostInfo:
    """Data class to store host information."""
    def __init__(self, hostname: str):
        self.hostname = hostname
        self.max_cores = 0
        self.running_cores = 0
        self.total_mem = ""
        self.reserved_mem = ""
    
    def __repr__(self):
        return (f"HostInfo(hostname='{self.hostname}', max_cores={self.max_cores}, "
                f"running_cores={self.running_cores}, total_mem='{self.total_mem}', "
                f"reserved_mem='{self.reserved_mem}')")


def parse_memory_value(mem_str: str) -> str:
    """Extract memory value from a string (handles values like '3.8G', '0M', etc.)"""
    match = re.search(r'[\d.]+[KMGT]', mem_str)
    return match.group(0) if match else ""


def parse_bhosts_output(output: str) -> List[HostInfo]:
    """Parse the bhosts -l output and extract host information."""
    hosts = []
    current_host = None
    in_load_section = False
    
    lines = output.strip().split('\n')
    
    for line in lines:
        if line.startswith('HOST  '):
            # Extract hostname from the line
            parts = line.split()
            if len(parts) >= 2:
                hostname = parts[1]
                current_host = HostInfo(hostname)
                hosts.append(current_host)
                in_load_section = False
        
        # Check for the STATUS line to extract MAX and RUN
        elif line.startswith('STATUS') and current_host:
            continue
        elif current_host and 'CPUF' in line and 'MAX' in line:
            # Skip header line
            continue
        elif current_host and re.match(r'^(closed_Full|ok)\s+', line):
            # This line contains STATUS, CPUF, JL/U, MAX, NJOBS, RUN, etc.
            parts = line.split()
            if len(parts) >= 6:
                try:
                    current_host.max_cores = int(parts[3])  # MAX column
                    current_host.running_cores = int(parts[5])  # RUN column
                except (ValueError, IndexError):
                    pass
        
        elif 'CURRENT LOAD USED FOR SCHEDULING:' in line:
            in_load_section = True
        
        # Parse memory information from the load section
        elif in_load_section and current_host:
            # Look for Total and Reserved lines
            if line.strip().startswith('Total'):
                # Extract memory value from the mem column
                parts = line.split()
                if len(parts) >= 12:
                    current_host.total_mem = parse_memory_value(parts[11])
            
            elif line.strip().startswith('Reserved'):
                # Extract memory value from the mem column
                parts = line.split()
                if len(parts) >= 12:
                    current_host.reserved_mem = parse_memory_value(parts[11])
        
        # Check for end of load section
        elif 'LOAD THRESHOLD USED FOR SCHEDULING:' in line:
            in_load_section = False
    
    return hosts


def run_bhosts_command() -> Optional[str]:
    """Execute bhosts -l command and return output."""
    try:
        result = subprocess.run(['bhosts', '-l'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running bhosts command: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("bhosts command not found. Make sure LSF is installed and in PATH.")
        return None


def get_hosts_info() -> Dict[str, Dict[str, any]]:
    output = run_bhosts_command()
    
    if output:
        hosts = parse_bhosts_output(output)
        hosts_dict = {
            host.hostname: {
                'max_cores': host.max_cores,
                'running_cores': host.running_cores,
                'total_mem': host.total_mem,
                'reserved_mem': host.reserved_mem
            }
            for host in hosts
        }
        
        return hosts_dict
    return {}


def main():
    hosts_dict = get_hosts_info()
    
    if hosts_dict:
        import json
        print(json.dumps(hosts_dict, indent=2))


if __name__ == "__main__":
    main()