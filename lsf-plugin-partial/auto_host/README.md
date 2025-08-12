# LSF AUTO_HOST ML Plugin

This plugin is designed for IBM Platform LSF Version 10.1 and enables machine learning-based host selection via the AUTO_HOST extension and a RESTful ML inference service.

## Contents

1. [In this Directory](#in-this-directory)
2. [About This Plugin](#about-this-plugin)
3. [Building and Installing the Plugin](#building-and-installing-the-plugin)
4. [Enabling the Plugin in LSF](#enabling-the-plugin-in-lsf)

---

## 1. In this Directory

- **README.md**: This file.
- **Makefile**: Build instructions for the plugin.
- **auto_host.c**: Source code for the ML-based AUTO_HOST plugin.

---

## 2. About This Plugin

The AUTO_HOST plugin allows LSF to leverage machine learning for host selection. Jobs can be submitted directly using the `bsub` command with the `-ext "AUTO_HOST[]"` option.

### Example Job Submission

```bash
bsub -n 3 -R "rusage[mem=222]" -R "type==any" -sim "runtime=60 cputime=50" -ext "AUTO_HOST[]" sleep 1000
```

For more details on the ML service, refer to the `lsf-ml-service` README.

---

## 3. Building and Installing the Plugin

1. **Clean previous builds (optional):**
   ```bash
   make clean
   ```
   *(If this is your first build, you can skip this step.)*

2. **Build the plugin:**
   ```bash
   make
   ```

3. **Copy the built plugin to the LSF lib directory:**
   ```bash
   cp schmod_auto_host.so /opt/ibm/lsf/10.1/linux3.10-glibc2.17-x86_64/lib
   ```
   *(The plugin should be built under `/opt/ibm/lsf/10.1/misc/examples/`.)*

---

## 4. Enabling the Plugin in LSF

1. **Edit your LSF configuration (lsf.conf or lsb.modules):**
   ```
   schmod_auto_host          ()           ()
   ```

2. **Restart the LSF scheduler:**
   ```bash
   badmin mbdrestart
   ```

---

## Additional Notes

- **Host Priorities:** The plugin uses metrics like throughput and job completion rates to determine host priorities.
- **Reward Metrics:** Metrics such as makespan and fairness are considered for prioritization.
