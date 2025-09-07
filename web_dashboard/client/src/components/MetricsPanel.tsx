import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  LinearProgress,
  Chip
} from '@mui/material';
import { 
  Schedule,
  Assignment,
  CheckCircle,
  TrendingUp
} from '@mui/icons-material';

interface Metrics {
  total_jobs: number;
  completed_jobs: number;
  active_jobs: number;
  decisions_made: number;
  completion_rate: number;
}

interface Props {
  metrics: Metrics;
  time: number;
  maxTime: number;
}

const MetricsPanel: React.FC<Props> = ({ metrics, time, maxTime }) => {
  // Debug: log the metrics data
  console.log('MetricsPanel received:', { metrics, time, maxTime });
  
  // Ensure all metrics have default values
  const safeMetrics = {
    total_jobs: metrics?.total_jobs || 0,
    completed_jobs: metrics?.completed_jobs || 0,
    active_jobs: metrics?.active_jobs || 0,
    decisions_made: metrics?.decisions_made || 0,
    completion_rate: metrics?.completion_rate || 0
  };
  
  const timeProgress = maxTime > 0 ? (time / maxTime) * 100 : 0;
  
  const MetricCard: React.FC<{
    title: string;
    value: string | number;
    subtitle?: string;
    icon: React.ReactNode;
    color?: string;
    progress?: number;
  }> = ({ title, value, subtitle, icon, color = '#2196f3', progress }) => (
    <Card className="metrics-card" elevation={2}>
      <CardContent sx={{ p: 2 }}>
        <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
          <Box sx={{ color, mr: 1 }}>{icon}</Box>
          <Typography variant="body2" color="text.secondary">
            {title}
          </Typography>
        </Box>
        <Typography variant="h4" className="metric-value" sx={{ color }}>
          {value}
        </Typography>
        {subtitle && (
          <Typography variant="caption" color="text.secondary">
            {subtitle}
          </Typography>
        )}
        {progress !== undefined && (
          <Box mt={1}>
            <LinearProgress 
              variant="determinate" 
              value={progress} 
              sx={{ 
                height: 6, 
                borderRadius: 3,
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
      <MetricCard
        title="Episode Time"
        value={`${time}s`}
        subtitle={time <= maxTime ? `Arrivals: ${time}/${maxTime}s` : `Completion: +${time - maxTime}s`}
        icon={<Schedule />}
        color={time <= maxTime ? "#ff9800" : "#9c27b0"}
        progress={time <= maxTime ? timeProgress : undefined}
      />
      
      <MetricCard
        title="Total Jobs"
        value={safeMetrics.total_jobs}
        icon={<Assignment />}
        color="#9c27b0"
      />
      
      <MetricCard
        title="Completed"
        value={safeMetrics.completed_jobs}
        subtitle={`${safeMetrics.completion_rate.toFixed(1)}% complete`}
        icon={<CheckCircle />}
        color="#4caf50"
        progress={safeMetrics.completion_rate}
      />
      
      <MetricCard
        title="Active Jobs"
        value={safeMetrics.active_jobs}
        icon={<TrendingUp />}
        color="#f44336"
      />
      
      <MetricCard
        title="Decisions Made"
        value={safeMetrics.decisions_made}
        icon={<Assignment />}
        color="#2196f3"
      />

      <Box mt={2}>
        <Typography variant="subtitle2" gutterBottom color="text.secondary">
          Episode Status
        </Typography>
        <Chip 
          label={time < maxTime ? 'Running' : 'Complete'}
          color={time < maxTime ? 'primary' : 'success'}
          size="small"
          sx={{ mb: 1 }}
        />
        
        {safeMetrics.total_jobs > 0 && (
          <Box>
            <Typography variant="caption" color="text.secondary">
              Efficiency: {(safeMetrics.completed_jobs / Math.max(safeMetrics.decisions_made, 1) * 100).toFixed(1)}%
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default MetricsPanel;