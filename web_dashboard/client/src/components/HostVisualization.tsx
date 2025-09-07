import React from 'react';
import { Box } from '@mui/material';

interface Host {
  id: number;
  cpu_util: number;
  memory_util: number;
  total_cores?: number;
  total_memory?: number;
}

interface HostConfig {
  host_id: number;
  total_cores: number;
  total_memory: number;
}

interface Props {
  hosts: Host[];
  hostConfigs: HostConfig[];
}

const HostVisualization: React.FC<Props> = ({ hosts, hostConfigs }) => {
  const getUtilizationClass = (cpuUtil: number, memoryUtil: number) => {
    const maxUtil = Math.max(cpuUtil, memoryUtil);
    if (maxUtil > 80) return 'high-utilization';
    if (maxUtil > 50) return 'medium-utilization';
    if (maxUtil > 10) return 'low-utilization';
    return 'idle';
  };

  const getGridCols = (hostCount: number) => {
    if (hostCount <= 10) return 5;
    if (hostCount <= 25) return 5;
    if (hostCount <= 50) return 8;
    return 10;
  };

  const gridCols = getGridCols(hosts.length);

  return (
    <Box 
      className="host-grid" 
      sx={{ 
        gridTemplateColumns: `repeat(${gridCols}, 1fr)`,
        maxHeight: '540px',
        overflowY: 'auto'
      }}
    >
      {hosts.map((host, index) => {
        const config = hostConfigs[host.id] || hostConfigs[index];
        const utilizationClass = getUtilizationClass(host.cpu_util, host.memory_util);
        
        return (
          <Box
            key={host.id}
            className={`host-card ${utilizationClass}`}
            sx={{ minHeight: '80px' }}
          >
            <div className="host-id">Host {host.id}</div>
            {config && (
              <div className="host-specs">
                {config.total_cores}c / {Math.round(config.total_memory / 1024)}GB
              </div>
            )}
            
            <div className="utilization-bars">
              <div>
                <div className="util-label">CPU: {host.cpu_util.toFixed(1)}%</div>
                <div className="util-bar">
                  <div 
                    className="util-fill cpu-fill" 
                    style={{ width: `${Math.min(host.cpu_util, 100)}%` }}
                  />
                </div>
              </div>
              
              <div>
                <div className="util-label">Mem: {host.memory_util.toFixed(1)}%</div>
                <div className="util-bar">
                  <div 
                    className="util-fill memory-fill" 
                    style={{ width: `${Math.min(host.memory_util, 100)}%` }}
                  />
                </div>
              </div>
            </div>
          </Box>
        );
      })}
    </Box>
  );
};

export default HostVisualization;