import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

interface EpisodeData {
  visualizer_step?: number;
  env_time?: number;
  time?: number;
  hosts: Array<{
    id: number;
    cpu_util: number;
    memory_util: number;
  }>;
  env_data?: {
    total_jobs: number;
    completed_jobs: number;
    failed_jobs: number;
    active_jobs: number;
    job_queue_length: number;
    needs_decision: boolean;
    episode_done: boolean;
  };
  metrics?: {
    total_jobs: number;
    completed_jobs: number;
    active_jobs: number;
    decisions_made: number;
    completion_rate: number;
  };
}

interface Props {
  episodeData: EpisodeData;
  isPaused?: boolean;
}

interface ChartData {
  time: number[];
  avgCpu: number[];
  avgMemory: number[];
  activeJobs: number[];
}

const ResourceChart: React.FC<Props> = ({ episodeData, isPaused = false }) => {
  const [chartData, setChartData] = useState<ChartData>({
    time: [],
    avgCpu: [],
    avgMemory: [],
    activeJobs: []
  });

  useEffect(() => {
    if (episodeData.hosts.length > 0 && !isPaused) {
      const avgCpu = episodeData.hosts.reduce((sum, host) => sum + host.cpu_util, 0) / episodeData.hosts.length;
      const avgMemory = episodeData.hosts.reduce((sum, host) => sum + host.memory_util, 0) / episodeData.hosts.length;
      const currentTime = episodeData.env_time || episodeData.time || 0;
      const activeJobs = episodeData.env_data?.active_jobs || episodeData.metrics?.active_jobs || 0;
      
      setChartData(prev => {
        const newTime = [...prev.time, currentTime];
        const newAvgCpu = [...prev.avgCpu, avgCpu];
        const newAvgMemory = [...prev.avgMemory, avgMemory];
        const newActiveJobs = [...prev.activeJobs, activeJobs];

        // Keep only last 200 points for better history
        const maxPoints = 200;
        return {
          time: newTime.slice(-maxPoints),
          avgCpu: newAvgCpu.slice(-maxPoints),
          avgMemory: newAvgMemory.slice(-maxPoints),
          activeJobs: newActiveJobs.slice(-maxPoints)
        };
      });
    }
  }, [episodeData, isPaused]);

  const plotData = [
    {
      x: chartData.time,
      y: chartData.avgCpu,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Average CPU %',
      line: { color: '#f44336', width: 2 },
      yaxis: 'y'
    },
    {
      x: chartData.time,
      y: chartData.avgMemory,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Average Memory %',
      line: { color: '#2196f3', width: 2 },
      yaxis: 'y'
    },
    {
      x: chartData.time,
      y: chartData.activeJobs,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Active Jobs',
      line: { color: '#4caf50', width: 2 },
      yaxis: 'y2'
    }
  ];

  const layout = {
    height: 320,
    margin: { t: 30, r: 80, b: 80, l: 80 },
    showlegend: true,
    legend: { 
      orientation: 'h' as const, 
      x: 0.5,
      xanchor: 'center' as const,
      y: -0.3,
      bgcolor: 'rgba(255,255,255,0.8)',
      bordercolor: '#e0e0e0',
      borderwidth: 1
    },
    xaxis: {
      title: {
        text: 'Simulation Time (seconds)',
        standoff: 20
      },
      gridcolor: '#e0e0e0',
      showgrid: true
    },
    yaxis: {
      title: {
        text: 'Resource Utilization (%)',
        standoff: 30
      },
      side: 'left' as const,
      range: [0, 100],
      gridcolor: '#e0e0e0',
      showgrid: true
    },
    yaxis2: {
      title: {
        text: 'Active Jobs Count',
        standoff: 30
      },
      side: 'right' as const,
      overlaying: 'y' as const,
      gridcolor: 'transparent',
      showgrid: false
    },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: 'white',
    font: {
      size: 12,
      family: 'Arial, sans-serif'
    }
  };

  const config = {
    displayModeBar: false,
    responsive: true
  };

  return (
    <div className="chart-container">
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};

export default ResourceChart;