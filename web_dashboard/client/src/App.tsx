import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Paper,
  Grid,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Chip,
  LinearProgress,
  Alert,
  AppBar,
  Toolbar
} from '@mui/material';
import HostVisualization from './components/HostVisualization';
import ResourceChart from './components/ResourceChart';
import JobQueueVisualization from './components/JobQueueVisualization';
import { useSocket } from './hooks/useSocket';
import { ConfigService } from './services/ConfigService';
import './App.css';

interface Config {
  num_hosts: number;
  max_time: number;
  max_jobs_per_step: number;
  host_configs: Array<{ host_id: number; total_cores: number; total_memory: number }>;
  checkpoint_path: string;
  available_seeds: string[];
}

interface EpisodeData {
  visualizer_step?: number;
  env_time?: number;
  time?: number; // Backwards compatibility
  hosts: Array<{
    id: number;
    cpu_util: number;
    memory_util: number;
    total_cores?: number;
    total_memory?: number;
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
    jobs_arriving_this_second?: number;
    in_arrival_phase?: boolean;
    in_completion_phase?: boolean;
    job_arrival_end?: number;
    estimated_total_duration?: number;
  };
  decisions?: Array<any>;
}

const App: React.FC = () => {
  const [config, setConfig] = useState<Config | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [selectedSeed, setSelectedSeed] = useState('42');
  const [error, setError] = useState<string>('');
  const logDir = process.env.REACT_APP_LOG_DIR || 'logs/exp4';
  const [episodeData, setEpisodeData] = useState<EpisodeData>({
    visualizer_step: 0,
    env_time: 0,
    time: 0,
    hosts: [],
    env_data: {
      total_jobs: 0,
      completed_jobs: 0,
      failed_jobs: 0,
      active_jobs: 0,
      job_queue_length: 0,
      needs_decision: false,
      episode_done: false
    },
    metrics: {
      total_jobs: 0,
      completed_jobs: 0,
      active_jobs: 0,
      decisions_made: 0,
      completion_rate: 0
    }
  });

  const socket = useSocket('http://localhost:5001');

  const loadConfig = async () => {
    try {
      const configData = await ConfigService.loadConfig(logDir);
      setConfig(configData);
      setError('');
    } catch (err) {
      setError(`Failed to load configuration: ${err}`);
    }
  };

  useEffect(() => {
    loadConfig();
  }, []);

  useEffect(() => {
    if (socket) {
      socket.on('episode_update', (data: EpisodeData) => {
        console.log('Received episode update:', data);
        setEpisodeData(data);
      });


      socket.on('episode_complete', () => {
        setIsRunning(false);
      });

      socket.on('episode_error', (data: { error: string }) => {
        setError(data.error);
        setIsRunning(false);
      });

      return () => {
        socket.off('episode_update');
        socket.off('episode_complete');
        socket.off('episode_error');
      };
    }
  }, [socket]);

  const startEpisode = async () => {
    if (!config) return;

    try {
      setError('');
      setIsRunning(true);
      setIsPaused(false);
      
      await ConfigService.startEpisode(config, parseInt(selectedSeed));
    } catch (err) {
      setError(`Failed to start episode: ${err}`);
      setIsRunning(false);
    }
  };

  const pauseEpisode = async () => {
    try {
      const response = await ConfigService.pauseEpisode();
      setIsPaused(response.isPaused);
    } catch (err) {
      setError(`Failed to pause episode: ${err}`);
    }
  };

  const stopEpisode = async () => {
    try {
      await ConfigService.stopEpisode();
      setIsRunning(false);
      setIsPaused(false);
    } catch (err) {
      setError(`Failed to stop episode: ${err}`);
    }
  };

  if (!config) {
    return (
      <Container maxWidth="md" style={{ marginTop: '2rem' }}>
        <Paper elevation={3} style={{ padding: '2rem' }}>
          <Typography variant="h5" gutterBottom>
            Loading LSF Scheduler Configuration...
          </Typography>
          {error && <Alert severity="error">{error}</Alert>}
          <LinearProgress />
        </Paper>
      </Container>
    );
  }

  const currentTime = episodeData.env_time || episodeData.time || 0;

  return (
    <Box className="app">
      <AppBar position="static" sx={{ background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)' }}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            LSF Scheduler Visualization Dashboard
          </Typography>
          <Chip 
            label={`${config.num_hosts} Hosts`} 
            color="secondary" 
            sx={{ mr: 2 }}
          />
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 3, mb: 3, backgroundColor: 'transparent' }}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
            {error}
          </Alert>
        )}

        {/* Control Panel */}
        <Paper elevation={3} sx={{ p: 2, mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item>
              <Button
                variant="contained"
                color="primary"
                onClick={startEpisode}
                disabled={isRunning}
                size="large"
              >
                Start Episode
              </Button>
            </Grid>
            <Grid item>
              <Button
                variant="contained"
                color={isPaused ? "primary" : "warning"}
                onClick={pauseEpisode}
                disabled={!isRunning}
                size="large"
              >
                {isPaused ? 'Resume' : 'Pause'}
              </Button>
            </Grid>
            <Grid item>
              <Button
                variant="contained"
                color="error"
                onClick={stopEpisode}
                disabled={!isRunning}
                size="large"
              >
                Stop Episode
              </Button>
            </Grid>
            <Grid item>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Seed</InputLabel>
                <Select
                  value={selectedSeed}
                  label="Seed"
                  onChange={(e) => setSelectedSeed(e.target.value)}
                  disabled={isRunning}
                >
                  {config.available_seeds.map(seed => (
                    <MenuItem key={seed} value={seed}>Seed {seed}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs>
              <Box>
                <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 'bold' }}>
                  Step #{episodeData.visualizer_step || 0} • Time: {currentTime}s • Phase: {(episodeData.env_time || 0) < config.max_time 
                    ? `Job Arrivals (${Math.max(0, config.max_time - currentTime)}s remaining)`
                    : `Completion Phase (+${currentTime - config.max_time}s)`
                  }
                </Typography>
                
                {/* Episode Progress */}
                <Box mt={1} mb={1}>
                  <LinearProgress
                    variant="determinate"
                    value={episodeData.env_data?.total_jobs ? 
                      (episodeData.env_data.completed_jobs / episodeData.env_data.total_jobs) * 100 : 0
                    }
                    sx={{
                      height: 8,
                      borderRadius: 4,
                      backgroundColor: '#e0e0e0',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: '#4caf50'
                      }
                    }}
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                    {episodeData.env_data?.completed_jobs || 0} of {episodeData.env_data?.total_jobs || 0} jobs completed 
                    ({episodeData.env_data && episodeData.env_data.total_jobs ? 
                      ((episodeData.env_data.completed_jobs / episodeData.env_data.total_jobs) * 100).toFixed(1) : 0
                    }%)
                    {(episodeData.env_data?.active_jobs || 0) > 0 && ` • ${episodeData.env_data?.active_jobs} running`}
                  </Typography>
                </Box>
              </Box>
            </Grid>
            <Grid item>
              <Chip 
                label={isRunning ? 'Running' : 'Idle'} 
                color={isRunning ? 'success' : 'default'}
                variant={isRunning ? 'filled' : 'outlined'}
              />
            </Grid>
          </Grid>
        </Paper>

        <Grid container spacing={3}>
          {/* Cluster Host Utilizations - Full Width, Large */}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2, height: '600px' }}>
              <Typography variant="h6" gutterBottom>
                Cluster Host Utilizations
              </Typography>
              <HostVisualization 
                hosts={episodeData.hosts}
                hostConfigs={config.host_configs}
              />
            </Paper>
          </Grid>

          {/* Resource Utilization Chart - Full Width Row */}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2, height: '350px' }}>
              <Typography variant="h6" gutterBottom>
                Resource Utilization Over Time
              </Typography>
              <ResourceChart episodeData={episodeData} isPaused={isPaused} />
            </Paper>
          </Grid>

          {/* Job Queue Status - Full Width */}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2, maxHeight: '400px', overflow: 'auto', backgroundColor: 'white', position: 'relative', zIndex: 1 }}>
              <JobQueueVisualization episodeData={episodeData} />
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default App;