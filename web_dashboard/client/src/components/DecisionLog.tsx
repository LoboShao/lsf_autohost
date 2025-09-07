import React, { useEffect, useRef } from 'react';
import { Box, Typography } from '@mui/material';

interface Decision {
  time: number;
  job_cores_norm: number;
  job_memory_norm: number;
  selected_host: number;
  priority_score: number;
}

interface Props {
  decisions: Decision[];
  currentTime: number;
}

const DecisionLog: React.FC<Props> = ({ decisions, currentTime }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [decisions]);

  const formatDecision = (decision: Decision) => {
    const coresEst = Math.round(decision.job_cores_norm * 4); // Assuming max 4 cores
    const memoryEst = Math.round(decision.job_memory_norm * 4096); // Assuming max 4GB
    const isRecent = currentTime - decision.time < 5;
    
    return (
      <div 
        key={`${decision.time}-${decision.selected_host}`}
        className={`decision-entry ${isRecent ? 'recent' : ''}`}
      >
        <div style={{ fontWeight: 'bold', color: '#4fc3f7' }}>
          T={decision.time}s
        </div>
        <div>
          Job({coresEst}c, {memoryEst}MB) → Host {decision.selected_host}
        </div>
        <div style={{ fontSize: '10px', opacity: 0.8 }}>
          Priority: {decision.priority_score.toFixed(3)}
        </div>
      </div>
    );
  };

  if (decisions.length === 0) {
    return (
      <Box className="decision-log" ref={scrollRef}>
        <Typography variant="body2" style={{ color: '#888', fontStyle: 'italic' }}>
          Waiting for scheduling decisions...
        </Typography>
      </Box>
    );
  }

  return (
    <Box className="decision-log" ref={scrollRef}>
      {decisions.map(formatDecision)}
    </Box>
  );
};

export default DecisionLog;