# tinyML_edgeML_RP2040

A CMake-based firmware template for deploying TinyML and Edge ML models on the Raspberry Pi Pico (RP2040) microcontroller. The project provides a structured starting point for integrating TensorFlow Lite Micro models into RP2040 firmware, including a DSP preprocessing pipeline and a Bonsai classifier.

## Features

- CMake project template ready for RP2040 / Raspberry Pi Pico SDK
- TensorFlow Lite Micro model inference (`tflite_model.h`)
- DSP preprocessing pipeline (`dsp_pipeline.cpp/.h`)
- ML model wrapper (`ml_model.cpp/.h`)
- Bonsai decision tree classifier integration (`conv_bonsai.cpp`)
- Pre-compiled `.uf2` firmware binary in `build/` directory for direct flashing
- Pico SDK import via `pico_sdk_import.cmake`

## Tech Stack

- C / C++
- CMake
- Raspberry Pi Pico SDK
- TensorFlow Lite Micro

## Project Structure

| File / Directory | Description |
|---|---|
| `src/main.cpp` | Firmware entry point |
| `src/dsp_pipeline.cpp/.h` | DSP preprocessing pipeline |
| `src/ml_model.cpp/.h` | ML model inference wrapper |
| `src/tflite_model.h` | TFLite model flatbuffer header |
| `src/conv_bonsai.cpp` | Bonsai classifier integration |
| `build/` | Pre-compiled firmware outputs (`.uf2` file) |
| `lib/` | External libraries |
| `CMakeLists.txt` | CMake build configuration |
| `pico_sdk_import.cmake` | Raspberry Pi Pico SDK import helper |

## Requirements

- Raspberry Pi Pico SDK
- CMake 3.13+
- ARM GCC toolchain (`arm-none-eabi-gcc`)

## Setup

```bash
mkdir build && cd build
cmake ..
make
```

Flash `build/*.uf2` to the RP2040 by holding BOOTSEL while connecting USB, then copying the file to the mounted drive.

## License

MIT
