import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  Stack,
  LinearProgress,
  Divider
} from '@mui/material';

interface EpisodeData {
  visualizer_step?: number;
  env_time?: number;
  time?: number;
  env_data?: {
    total_jobs: number;
    completed_jobs: number;
    active_jobs: number;
    job_queue_length: number;
    deferred_jobs_length?: number;
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
    deferred_jobs_count?: number;
  };
}

interface Props {
  episodeData: EpisodeData;
}

const JobQueueVisualization: React.FC<Props> = ({ episodeData }) => {
  // Extract data with fallbacks
  const envData = episodeData.env_data;
  const metrics = episodeData.metrics;
  const currentTime = episodeData.env_time || episodeData.time || 0;
  const visualizerStep = episodeData.visualizer_step || 0;

  // Job counts
  const totalJobs = envData?.total_jobs || metrics?.total_jobs || 0;
  const completedJobs = envData?.completed_jobs || metrics?.completed_jobs || 0;
  const activeJobs = envData?.active_jobs || metrics?.active_jobs || 0;
  
  // Queue lengths
  const jobQueueLength = envData?.job_queue_length || 0;
  
  // Derived metrics
  const pendingJobs = totalJobs - completedJobs - activeJobs;
  const completionRate = totalJobs > 0 ? (completedJobs / totalJobs) * 100 : 0;
  
  // Phase information
  const arrivalPhase = metrics?.in_arrival_phase ?? (currentTime < (metrics?.job_arrival_end || 200));

  // Status indicators
  const isEpisodeRunning = !envData?.episode_done;

  const QueueMetricCard: React.FC<{
    title: string;
    value: number;
    subtitle?: string;
    color?: string;
    progress?: number;
  }> = ({ title, value, subtitle, color = '#2196f3', progress }) => (
    <Card elevation={1} sx={{ minHeight: 50 }}>
      <CardContent sx={{ p: 0.8, textAlign: 'center' }}>
        <Typography variant="caption" color="text.secondary" noWrap sx={{ fontSize: '0.7rem', mb: 0.5 }}>
          {title}
        </Typography>
        <Typography variant="h6" sx={{ color, fontWeight: 'bold', fontSize: '1rem' }}>
          {value}
        </Typography>
        {subtitle && (
          <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
            {subtitle}
          </Typography>
        )}
        {progress !== undefined && (
          <Box mt={0.5}>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{
                height: 3,
                borderRadius: 2,
                backgroundColor: '#e0e0e0',
                '& .MuiLinearProgress-bar': { backgroundColor: color }
              }}
            />
          </Box>
        )}
      </CardContent>
    </Card>
  );

  return (
    <Box>
      {/* Episode Status Header */}
      <Box mb={2}>
        <Typography variant="h6" gutterBottom>
          Job Queue & Status
        </Typography>
        
        <Stack direction="row" spacing={1} justifyContent="space-between" flexWrap="wrap">
          <Chip 
            size="small" 
            label={`Step ${visualizerStep}`} 
            color="primary" 
            variant="outlined"
          />
          <Chip 
            size="small" 
            label={`Time ${currentTime}s`} 
            color="secondary" 
            variant="outlined"
          />
          <Chip 
            size="small"
            label={arrivalPhase ? "Arrival Phase" : "Completion Phase"} 
            color={arrivalPhase ? "primary" : "success"}
            variant="filled"
          />
          <Chip 
            size="small"
            label={isEpisodeRunning ? "Running" : "Complete"} 
            color={isEpisodeRunning ? "secondary" : "success"}
            variant="outlined"
          />
        </Stack>
      </Box>

      <Divider sx={{ my: 1 }} />

      {/* Two Column Layout: Job Cards Left, Queue Bars Right */}
      <Box display="grid" gridTemplateColumns="1fr 1fr" gap={2}>
        
        {/* Left Column: Job Status Cards */}
        <Box>
          <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
            Job Status
          </Typography>
          <Box display="grid" gridTemplateColumns="1fr" gap={1}>
            <QueueMetricCard
              title="Total Jobs"
              value={totalJobs}
              color="#2196f3"
            />
            
            <QueueMetricCard
              title="Completed"
              value={completedJobs}
              subtitle={`${completionRate.toFixed(1)}% done`}
              color="#4caf50"
              progress={completionRate}
            />
            
            <QueueMetricCard
              title="Active Jobs"
              value={activeJobs}
              subtitle="Currently running"
              color="#ff9800"
            />
          </Box>
        </Box>

        {/* Right Column: Queue Status Bars */}
        <Box>
          <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
            Queue Status
          </Typography>
          
          <Box>
            {/* Job Queue Bar */}
            <Box mb={1.5}>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="text.secondary">Job Queue</Typography>
                <Typography variant="body2" sx={{ fontWeight: 'bold', color: '#2196f3' }}>
                  {jobQueueLength}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={Math.min((jobQueueLength / Math.max(totalJobs, 1)) * 100, 100)}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: '#e3f2fd',
                  '& .MuiLinearProgress-bar': { backgroundColor: '#2196f3', borderRadius: 4 }
                }}
              />
              <Typography variant="caption" color="text.secondary">
                Jobs waiting to run
              </Typography>
            </Box>

            {/* Deferred Jobs Bar */}
            <Box mb={1.5}>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="text.secondary">Deferred Jobs</Typography>
                <Typography variant="body2" sx={{ fontWeight: 'bold', color: '#ff9800' }}>
                  {(envData?.deferred_jobs_length || metrics?.deferred_jobs_count || 0)}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={Math.min(((envData?.deferred_jobs_length || metrics?.deferred_jobs_count || 0) / Math.max(totalJobs, 1)) * 100, 100)}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: '#fff3e0',
                  '& .MuiLinearProgress-bar': { backgroundColor: '#ff9800', borderRadius: 4 }
                }}
              />
              <Typography variant="caption" color="text.secondary">
                Jobs deferred due to resources
              </Typography>
            </Box>

            {/* Pending Jobs Bar */}
            <Box mb={1}>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="text.secondary">Pending Jobs</Typography>
                <Typography variant="body2" sx={{ fontWeight: 'bold', color: '#607d8b' }}>
                  {Math.max(0, pendingJobs)}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={Math.min((Math.max(0, pendingJobs) / Math.max(totalJobs, 1)) * 100, 100)}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: '#f5f5f5',
                  '& .MuiLinearProgress-bar': { backgroundColor: '#607d8b', borderRadius: 4 }
                }}
              />
              <Typography variant="caption" color="text.secondary">
                Jobs not yet started
              </Typography>
            </Box>
          </Box>
        </Box>
      </Box>

    </Box>
  );
};

export default JobQueueVisualization;