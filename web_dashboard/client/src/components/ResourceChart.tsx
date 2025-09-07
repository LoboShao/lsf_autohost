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
}

const ResourceChart: React.FC<Props> = ({ episodeData, isPaused = false }) => {
  const [chartData, setChartData] = useState<ChartData>({
    time: [],
    avgCpu: [],
    avgMemory: []
  });

  useEffect(() => {
    if (episodeData.hosts.length > 0 && !isPaused) {
      const avgCpu = episodeData.hosts.reduce((sum, host) => sum + host.cpu_util, 0) / episodeData.hosts.length;
      const avgMemory = episodeData.hosts.reduce((sum, host) => sum + host.memory_util, 0) / episodeData.hosts.length;
      const currentTime = episodeData.env_time || episodeData.time || 0;
      
      setChartData(prev => {
        const newTime = [...prev.time, currentTime];
        const newAvgCpu = [...prev.avgCpu, avgCpu];
        const newAvgMemory = [...prev.avgMemory, avgMemory];

        // Keep only last 200 points for better performance
        const maxPoints = 200;
        return {
          time: newTime.slice(-maxPoints),
          avgCpu: newAvgCpu.slice(-maxPoints),
          avgMemory: newAvgMemory.slice(-maxPoints)
        };
      });
    }
  }, [episodeData, isPaused]);

  const plotData = [
    // CPU utilization - smooth line for trend visibility
    {
      x: chartData.time,
      y: chartData.avgCpu,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'CPU Utilization',
      line: { 
        color: '#2196F3', 
        width: 3,
        smoothing: 0.6
      },
      hovertemplate: '<b>CPU</b>: %{y:.1f}%<br>Time: %{x}s<extra></extra>'
    },
    // Memory utilization - smooth line for trend visibility  
    {
      x: chartData.time,
      y: chartData.avgMemory,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Memory Utilization',
      line: { 
        color: '#FF5722', 
        width: 3,
        smoothing: 0.6
      },
      hovertemplate: '<b>Memory</b>: %{y:.1f}%<br>Time: %{x}s<extra></extra>'
    }
  ];

  const layout = {
    title: {
      text: 'Resource Utilization Over Time',
      font: { size: 14, color: '#333' },
      x: 0.5,
      xanchor: 'center' as const
    },
    height: 320,
    margin: { t: 50, r: 50, b: 80, l: 70 },
    showlegend: true,
    legend: { 
      orientation: 'h' as const, 
      x: 0.5,
      xanchor: 'center' as const,
      y: -0.25,
      bgcolor: 'rgba(255,255,255,0.9)',
      bordercolor: 'rgba(200,200,200,0.5)',
      borderwidth: 1,
      font: { size: 12 }
    },
    xaxis: {
      title: {
        text: 'Simulation Time (seconds)',
        font: { size: 12 }
      },
      gridcolor: 'rgba(200,200,200,0.3)',
      showgrid: true,
      zeroline: false
    },
    yaxis: {
      title: {
        text: 'Utilization (%)',
        font: { size: 12 }
      },
      range: [0, 100],
      gridcolor: 'rgba(200,200,200,0.3)',
      showgrid: true,
      ticksuffix: '%',
      dtick: 20
    },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: 'white',
    hovermode: 'x unified' as const,
    font: {
      size: 11,
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