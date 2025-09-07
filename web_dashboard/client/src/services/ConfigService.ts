import axios from 'axios';

const API_BASE = process.env.NODE_ENV === 'production' 
  ? '' 
  : 'http://localhost:5001';

export class ConfigService {
  static async loadConfig(logDir: string) {
    try {
      const encodedLogDir = encodeURIComponent(logDir);
      const response = await axios.get(`${API_BASE}/api/config/${encodedLogDir}`);
      return response.data;
    } catch (error) {
      console.error('Failed to load config:', error);
      throw error;
    }
  }

  static async startEpisode(config: any, seed: number) {
    try {
      const response = await axios.post(`${API_BASE}/api/start-episode`, {
        config,
        seed
      });
      return response.data;
    } catch (error) {
      console.error('Failed to start episode:', error);
      throw error;
    }
  }

  static async pauseEpisode() {
    try {
      const response = await axios.post(`${API_BASE}/api/pause-episode`);
      return response.data;
    } catch (error) {
      console.error('Failed to pause episode:', error);
      throw error;
    }
  }

  static async stopEpisode() {
    try {
      const response = await axios.post(`${API_BASE}/api/stop-episode`);
      return response.data;
    } catch (error) {
      console.error('Failed to stop episode:', error);
      throw error;
    }
  }
}