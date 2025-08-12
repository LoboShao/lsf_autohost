# LSF ML Service

This folder contains the RESTful service and model for ML-based host selection in LSF environments.

## How to Use

1. **Start the RESTful Model Service**

   Run the following command to start the Flask-based REST API:
   
   ```bash
   python3 app.py
   ```
   
   This will start the service on port 5000 (by default) and keep it running in the foreground.

2. **LSF Plugin Integration**

   Once the service is running, the LSF plugin (deployed in your LSF installation) will automatically call this REST API for host selection during job scheduling.

3. **Example: Submitting a Job with AUTO_HOST**

   The following is an example of how to submit a job directly using the bsub command with AUTO_HOST extension:

   ```bash
   bsub -n 3 -R "rusage[mem=222]" -R "type==any" -sim "runtime=60 cputime=50" -ext "AUTO_HOST[]" sleep 1000
   ```

   This will submit a job to LSF with ML-based host selection enabled through the AUTO_HOST extension.

---