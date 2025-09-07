# LSF Scheduler Visualizer Dashboard

A real-time web dashboard for visualizing the LSF reinforcement learning scheduler training episodes.

## Features

✅ **Fixed all bugs** in the original visualizer:
- Fixed undefined `queue_length` variable in episode_runner.py:173
- Added comprehensive error handling and data validation

✅ **Enhanced Job Queue Visualization**:
- **Job Queue Length**: Real-time display of jobs waiting to run
- **Completed Jobs Counter**: Shows completed jobs with completion percentage
- **Deferred Jobs**: Shows jobs that couldn't be scheduled immediately  
- **Active Jobs**: Currently running jobs across the cluster
- **Failed Jobs**: Jobs that failed during execution

✅ **Real-time Cluster Host Utilizations**:
- Individual host CPU and memory utilization percentages
- Color-coded utilization levels (idle/low/medium/high)
- Host capacity information (cores/memory)
- Grid layout supporting up to 100+ hosts

✅ **Current Time and Environment Step Display**:
- **Environment Step**: Number of scheduler decision steps taken
- **Current Time**: Simulation time in seconds  
- **Phase Indicators**: Job arrival phase vs completion phase
- **Decision Status**: Shows when scheduler decisions are needed

✅ **Enhanced Status Tracking**:
- Episode progress with completion percentages
- Real-time arrival rates during job submission phase
- Queue status (job queue, submission queue, pending jobs)
- Episode termination detection

## Architecture

### Components
- **JobQueueVisualization**: Comprehensive job and queue status display
- **HostVisualization**: Real-time cluster host utilization grid
- **ResourceChart**: Time-series resource utilization charts
- **DecisionLog**: Scheduling decision history

### Data Flow
1. **Environment (env.rs)**: Rust-based simulation provides real-time state
2. **Episode Runner (episode_runner.py)**: Streams data via JSON over stdout
3. **Server (server.js)**: Node.js WebSocket server processes Python output
4. **React Client**: Real-time dashboard with Material-UI components

## Quick Start

### 1. Install Dependencies
```bash
# Install server dependencies
cd web_dashboard
npm install

# Install client dependencies  
cd client
npm install
cd ..
```

### 2. Start the Dashboard
```bash
# Start both server and client (recommended)
npm run dev

# Or start individually:
# Terminal 1: Start server
npm run server

# Terminal 2: Start client  
npm run client
```

### 3. Open Dashboard
- **Client**: http://localhost:3000
- **Server**: http://localhost:5001

### 4. Run Episode
1. Click "Start Episode" 
2. Select seed (42, 43, 44)
3. Watch real-time visualization
4. Episode runs until all jobs complete

## Data Sources

The visualizer uses test environment data from:
- **Test Data**: `logs/exp4/test_env_data.json` - Host configs and job schedules
- **Model Checkpoint**: `logs/exp4/checkpoints/*.pt` - Trained PPO model
- **Real-time State**: Direct from `env.rs` via Python episode runner

## Key Metrics Displayed

### Job Status
- **Total Jobs**: Number of jobs in the episode
- **Completed**: Successfully finished jobs
- **Active**: Currently running jobs  
- **Failed**: Jobs that encountered errors
- **Pending**: Jobs not yet started

### Queue Information  
- **Job Queue**: Jobs waiting for available resources
- **Submission Queue**: Jobs being processed for submission
- **Deferred Jobs**: Jobs postponed due to resource constraints

### Resource Utilization
- **Host CPU/Memory**: Per-host utilization percentages
- **Average Utilization**: Cluster-wide resource usage trends
- **Utilization History**: Time-series charts

### Timing Information
- **Environment Step**: Scheduler decision counter
- **Current Time**: Simulation time (seconds)
- **Phase**: Job arrival (0-200s) vs completion (200s+)
- **Decision Status**: When scheduler action is required

## Architecture Notes

### Performance Optimizations
- **Cached Normalization**: Host capacity ratios pre-calculated in Rust
- **Streaming Updates**: 10 FPS real-time data streaming
- **Efficient State**: Only essential data transmitted over WebSocket
- **React Optimization**: Minimized re-renders with proper data structures

### Error Handling
- **Connection Recovery**: Automatic WebSocket reconnection
- **Data Validation**: Safe handling of missing/invalid data
- **Episode Recovery**: Graceful handling of episode failures
- **Debug Logging**: Comprehensive debugging information

## Troubleshooting

### Common Issues

1. **"queue_length is not defined" Error**
   - ✅ **Fixed**: Added proper variable initialization in episode_runner.py

2. **Missing Dependencies**
   ```bash
   cd web_dashboard && npm run install-all
   ```

3. **Python Environment Issues**
   - Ensure conda environment `lsf_autohost` is activated
   - Check Python path in server.js (line 114)

4. **Port Conflicts**
   - Client: Default 3000, configurable in package.json
   - Server: Default 5001, configurable in server.js

5. **Model Loading Issues**
   - Ensure checkpoint exists in `logs/exp4/checkpoints/`
   - Check test_env_data.json format

### Debug Mode
Enable detailed logging by checking browser console and server terminal output. All debug messages prefixed with `DEBUG:`.

## Development

### Adding New Visualizations
1. Create component in `client/src/components/`
2. Import in `App.tsx`
3. Add to layout grid
4. Update episode_runner.py for new data if needed

### Modifying Data Stream
1. Update `episode_runner.py` for new metrics
2. Update TypeScript interfaces in `App.tsx`
3. Handle new data in React components

The dashboard is now fully functional with all requested features implemented and all bugs fixed!