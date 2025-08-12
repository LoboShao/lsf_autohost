# LSF AutoHost Project

This repository contains the LSF AutoHost project, which integrates machine learning techniques to enhance host selection and scheduling in IBM Platform LSF.

## Project Structure

- **`lsf-plugin-partial/`**: Contains the LSF plugin-related code.
- **`training/`**: Includes the model training code. The launch file is `training/lsf_train.py`.

## Getting Started

1. To start the training process, run the following command:
   ```bash
   python training/lsf_train.py
   ```

2. Use TensorBoard to observe the training progress. Start TensorBoard with:
   ```bash
   tensorboard --logdir=<log_directory>
   ```
   Replace `<log_directory>` with the path to your log files.

## LSF Plugin

The LSF plugin-related code is located in the `lsf-plugin-partial` folder. This plugin enables machine learning-based host selection and integrates with LSF's scheduling system.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
