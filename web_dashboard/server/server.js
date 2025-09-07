const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: ["http://localhost:3000", "http://localhost:3001"],
    methods: ["GET", "POST"]
  }
});

app.use(cors());
app.use(express.json());

// Serve static files from React build in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../client/build')));
}

class LSFVisualizationServer {
  constructor() {
    this.isRunning = false;
    this.isPaused = false;
    this.pythonProcess = null;
    this.currentEpisodeData = {
      time: 0,
      hosts: [],
      jobs: [],
      decisions: [],
      metrics: {}
    };
  }

  async loadConfig(logDir) {
    try {
      // Use environment variable if set, otherwise use provided logDir
      const actualLogDir = process.env.LSF_VISUALIZER_LOG_DIR || logDir;
      // Resolve path relative to project root (parent of web_dashboard)
      const projectRoot = path.join(__dirname, '../..');
      const testDataPath = path.join(projectRoot, actualLogDir, 'test_env_data.json');
      const testData = JSON.parse(fs.readFileSync(testDataPath, 'utf8'));
      
      // Find checkpoint
      const checkpointDir = path.join(projectRoot, actualLogDir, 'checkpoints');
      const checkpoints = fs.readdirSync(checkpointDir)
        .filter(file => file.endsWith('.pt'))
        .sort();
      
      // Use specified checkpoint or latest
      const specifiedCheckpoint = process.env.LSF_VISUALIZER_CHECKPOINT;
      let checkpointPath = null;
      
      if (specifiedCheckpoint) {
        const specifiedPath = path.join(checkpointDir, specifiedCheckpoint);
        if (fs.existsSync(specifiedPath)) {
          checkpointPath = specifiedPath;
        } else {
          console.warn(`Specified checkpoint not found: ${specifiedCheckpoint}, using latest`);
        }
      }
      
      if (!checkpointPath && checkpoints.length > 0) {
        // Sort by modification time to get the latest
        const checkpointFiles = checkpoints.map(file => {
          const fullPath = path.join(checkpointDir, file);
          return { file, mtime: fs.statSync(fullPath).mtime };
        }).sort((a, b) => b.mtime - a.mtime);
        
        checkpointPath = path.join(checkpointDir, checkpointFiles[0].file);
      }

      // Extract environment config
      const env42 = testData.test_environments['42'];
      const hosts = env42.hosts;
      const jobSchedule = env42.job_schedule;

      const config = {
        num_hosts: hosts.length,
        max_time: jobSchedule.max_time,
        max_jobs_per_step: jobSchedule.max_jobs_per_step,
        host_configs: hosts,
        checkpoint_path: checkpointPath,
        test_data_path: testDataPath,
        available_seeds: Object.keys(testData.test_environments)
      };

      return config;
    } catch (error) {
      console.error('Error loading config:', error);
      throw error;
    }
  }

  startEpisode(config, seed = 42) {
    if (this.isRunning) {
      return false;
    }

    this.isRunning = true;
    this.currentEpisodeData = {
      time: 0,
      hosts: new Array(config.num_hosts).fill(null).map((_, i) => ({
        id: i,
        cpu_util: 0,
        memory_util: 0,
        total_cores: config.host_configs[i].total_cores,
        total_memory: config.host_configs[i].total_memory
      })),
      jobs: [],
      decisions: [],
      metrics: {
        total_jobs: 0,
        completed_jobs: 0,
        active_jobs: 0,
        decisions_made: 0,
        completion_rate: 0
      }
    };

    // Start Python episode runner
    this.runPythonEpisode(config, seed);
    return true;
  }

  runPythonEpisode(config, seed) {
    // Use the simple Python episode runner
    const scriptPath = path.join(__dirname, 'simple_episode_runner.py');
    
    // Set Python path to use the conda environment
    const pythonPath = process.env.PYTHON_PATH || '/Users/yimingshao/miniconda3/envs/lsf_autohost/bin/python';
    
    // Run Python process using direct conda environment executable
    const projectRoot = path.join(__dirname, '../..');
    
    const logDir = process.env.LSF_VISUALIZER_LOG_DIR || 'logs/exp4';
    this.pythonProcess = spawn('bash', ['-c', 
      `"${pythonPath}" "${scriptPath}" --log-dir "${logDir}" --seed ${seed}`
    ], {
      cwd: projectRoot,
      env: { 
        ...process.env,
        PYTHONPATH: projectRoot,
        PATH: '/Users/yimingshao/miniconda3/envs/lsf_autohost/bin:' + process.env.PATH
      }
    });

    this.pythonProcess.stdout.on('data', (data) => {
      try {
        const lines = data.toString().split('\n').filter(line => line.trim());
        lines.forEach(line => {
          try {
            if (line.startsWith('EPISODE_DATA:')) {
              const jsonStr = line.substring(13);  // "EPISODE_DATA:" = 13 chars
              const episodeData = JSON.parse(jsonStr);
              this.processEpisodeUpdate(episodeData);
            } else if (line.startsWith('DECISION:')) {
              const jsonStr = line.substring(9);   // "DECISION:" = 9 chars
              const decision = JSON.parse(jsonStr);
              this.processDecision(decision);
            } else if (line.startsWith('ERROR:')) {
              const jsonStr = line.substring(6);    // "ERROR:" = 6 chars
              const error = JSON.parse(jsonStr);
              console.error('Python error:', error);
              io.emit('episode_error', { error: error.error });
            } else if (line.startsWith('EPISODE_START:')) {
              const jsonStr = line.substring(14);  // "EPISODE_START:" = 14 chars (not 15!)
              const startData = JSON.parse(jsonStr);
              console.log('Episode started:', startData);
              io.emit('episode_started', startData);
            } else if (line.startsWith('EPISODE_COMPLETE:')) {
              const jsonStr = line.substring(17); // "EPISODE_COMPLETE:" = 17 chars (not 18!)
              const completeData = JSON.parse(jsonStr);
              console.log('Episode completed:', completeData);
              io.emit('episode_complete', completeData);
            } else if (line.startsWith('DEBUG:')) {
              console.log('Python debug:', line.substring(6));
            } else {
              // Log unrecognized lines for debugging
              console.log('Unrecognized Python output:', line);
            }
          } catch (parseError) {
            console.error('JSON parse error for line:', line);
            console.error('Parse error:', parseError.message);
          }
        });
      } catch (error) {
        console.error('Error parsing Python output:', error);
      }
    });

    this.pythonProcess.stderr.on('data', (data) => {
      console.error('Python stderr:', data.toString());
    });

    this.pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      this.isRunning = false;
      io.emit('episode_complete');
    });

    this.pythonProcess.on('error', (error) => {
      console.error('Failed to start Python process:', error);
      this.isRunning = false;
      io.emit('episode_error', { error: error.message });
    });
  }


  processEpisodeUpdate(data) {
    // Handle both old and new data formats
    this.currentEpisodeData.time = data.env_time || data.time || 0;
    this.currentEpisodeData.visualizer_step = data.visualizer_step || 0;
    this.currentEpisodeData.env_time = data.env_time || data.time || 0;
    this.currentEpisodeData.hosts = data.hosts || [];
    this.currentEpisodeData.env_data = data.env_data || {};
    this.currentEpisodeData.metrics = data.metrics || data.env_data || {};
    
    // Only emit updates if not paused
    if (!this.isPaused) {
      io.emit('episode_update', {
        time: this.currentEpisodeData.time,
        visualizer_step: this.currentEpisodeData.visualizer_step,
        env_time: this.currentEpisodeData.env_time,
        hosts: this.currentEpisodeData.hosts,
        env_data: this.currentEpisodeData.env_data,
        metrics: this.currentEpisodeData.metrics
      });
    }
  }

  processDecision(decision) {
    this.currentEpisodeData.decisions.push(decision);
    
    // Keep only last 50 decisions
    if (this.currentEpisodeData.decisions.length > 50) {
      this.currentEpisodeData.decisions = this.currentEpisodeData.decisions.slice(-50);
    }
    
    io.emit('new_decision', decision);
  }

  pauseEpisode() {
    this.isPaused = !this.isPaused;
    return this.isPaused;
  }

  stopEpisode() {
    this.isRunning = false;
    this.isPaused = false;
    if (this.pythonProcess) {
      this.pythonProcess.kill();
      this.pythonProcess = null;
    }
  }
}

const visualizationServer = new LSFVisualizationServer();

// API Routes
app.get('/api/config/:logDir', async (req, res) => {
  try {
    const logDir = decodeURIComponent(req.params.logDir);
    const config = await visualizationServer.loadConfig(logDir);
    res.json(config);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/start-episode', (req, res) => {
  const { config, seed } = req.body;
  const success = visualizationServer.startEpisode(config, seed);
  res.json({ success });
});

app.post('/api/pause-episode', (req, res) => {
  const isPaused = visualizationServer.pauseEpisode();
  res.json({ success: true, isPaused });
});

app.post('/api/stop-episode', (req, res) => {
  visualizationServer.stopEpisode();
  res.json({ success: true });
});

// Socket.IO connections
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  // Send current episode data to new clients
  socket.emit('episode_update', visualizationServer.currentEpisodeData);
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Serve React app in production
if (process.env.NODE_ENV === 'production') {
  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../client/build', 'index.html'));
  });
}

const PORT = process.env.PORT || 5001;
server.listen(PORT, () => {
  console.log(`LSF Visualization Server running on port ${PORT}`);
});

module.exports = app;