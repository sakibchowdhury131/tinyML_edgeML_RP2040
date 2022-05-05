# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# compile ASM with /usr/bin/arm-none-eabi-gcc
# compile C with /usr/bin/arm-none-eabi-gcc
# compile CXX with /usr/bin/arm-none-eabi-g++
ASM_DEFINES = -DCMSIS_NN=1 -DCOMPILE_DEFINITIONS -DLIB_PICO_BIT_OPS=1 -DLIB_PICO_BIT_OPS_PICO=1 -DLIB_PICO_DIVIDER=1 -DLIB_PICO_DIVIDER_HARDWARE=1 -DLIB_PICO_DOUBLE=1 -DLIB_PICO_DOUBLE_PICO=1 -DLIB_PICO_FLOAT=1 -DLIB_PICO_FLOAT_PICO=1 -DLIB_PICO_INT64_OPS=1 -DLIB_PICO_INT64_OPS_PICO=1 -DLIB_PICO_MALLOC=1 -DLIB_PICO_MEM_OPS=1 -DLIB_PICO_MEM_OPS_PICO=1 -DLIB_PICO_PLATFORM=1 -DLIB_PICO_PRINTF=1 -DLIB_PICO_PRINTF_PICO=1 -DLIB_PICO_RUNTIME=1 -DLIB_PICO_STANDARD_LINK=1 -DLIB_PICO_STDIO=1 -DLIB_PICO_STDIO_UART=1 -DLIB_PICO_STDLIB=1 -DLIB_PICO_SYNC=1 -DLIB_PICO_SYNC_CORE=1 -DLIB_PICO_SYNC_CRITICAL_SECTION=1 -DLIB_PICO_SYNC_MUTEX=1 -DLIB_PICO_SYNC_SEM=1 -DLIB_PICO_TIME=1 -DLIB_PICO_UTIL=1 -DPICO_BOARD=\"pico\" -DPICO_BUILD=1 -DPICO_CMAKE_BUILD_TYPE=\"Release\" -DPICO_COPY_TO_RAM=0 -DPICO_CXX_ENABLE_EXCEPTIONS=0 -DPICO_NO_FLASH=0 -DPICO_NO_HARDWARE=0 -DPICO_ON_DEVICE=1 -DPICO_TARGET_NAME=\"kernel_svdf_test\" -DPICO_USE_BLOCKED_RAM=0 -DTF_LITE_DISABLE_X86_NEON=1 -DTF_LITE_STATIC_MEMORY=1

ASM_INCLUDES = -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/tests/kernel_svdf_test/../../tests/kernel_svdf_test -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/cmsis/CMSIS/DSP/Include -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/ruy -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/gemmlowp -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/kissfft -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/flatbuffers -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/cmsis/CMSIS/Core/Include -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/cmsis -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/flatbuffers/include -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/cmsis/CMSIS/NN/Include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_stdlib/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_gpio/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_base/include -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/generated/pico_base -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/boards/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_platform/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2040/hardware_regs/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_base/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2040/hardware_structs/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_claim/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_sync/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_uart/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_divider/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_time/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_timer/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_sync/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_util/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_runtime/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_clocks/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_irq/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_resets/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_pll/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_vreg/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_watchdog/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_xosc/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_printf/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_bootrom/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_bit_ops/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_divider/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_double/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_int64_ops/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_float/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_malloc/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/boot_stage2/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_binary_info/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_stdio/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_stdio_uart/include

ASM_FLAGS = -mcpu=cortex-m0plus -mthumb -O3 -DNDEBUG -nostdlib -ffunction-sections -fdata-sections

C_DEFINES = -DCMSIS_NN=1 -DCOMPILE_DEFINITIONS -DLIB_PICO_BIT_OPS=1 -DLIB_PICO_BIT_OPS_PICO=1 -DLIB_PICO_DIVIDER=1 -DLIB_PICO_DIVIDER_HARDWARE=1 -DLIB_PICO_DOUBLE=1 -DLIB_PICO_DOUBLE_PICO=1 -DLIB_PICO_FLOAT=1 -DLIB_PICO_FLOAT_PICO=1 -DLIB_PICO_INT64_OPS=1 -DLIB_PICO_INT64_OPS_PICO=1 -DLIB_PICO_MALLOC=1 -DLIB_PICO_MEM_OPS=1 -DLIB_PICO_MEM_OPS_PICO=1 -DLIB_PICO_PLATFORM=1 -DLIB_PICO_PRINTF=1 -DLIB_PICO_PRINTF_PICO=1 -DLIB_PICO_RUNTIME=1 -DLIB_PICO_STANDARD_LINK=1 -DLIB_PICO_STDIO=1 -DLIB_PICO_STDIO_UART=1 -DLIB_PICO_STDLIB=1 -DLIB_PICO_SYNC=1 -DLIB_PICO_SYNC_CORE=1 -DLIB_PICO_SYNC_CRITICAL_SECTION=1 -DLIB_PICO_SYNC_MUTEX=1 -DLIB_PICO_SYNC_SEM=1 -DLIB_PICO_TIME=1 -DLIB_PICO_UTIL=1 -DPICO_BOARD=\"pico\" -DPICO_BUILD=1 -DPICO_CMAKE_BUILD_TYPE=\"Release\" -DPICO_COPY_TO_RAM=0 -DPICO_CXX_ENABLE_EXCEPTIONS=0 -DPICO_NO_FLASH=0 -DPICO_NO_HARDWARE=0 -DPICO_ON_DEVICE=1 -DPICO_TARGET_NAME=\"kernel_svdf_test\" -DPICO_USE_BLOCKED_RAM=0 -DTF_LITE_DISABLE_X86_NEON=1 -DTF_LITE_STATIC_MEMORY=1

C_INCLUDES = -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/tests/kernel_svdf_test/../../tests/kernel_svdf_test -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/cmsis/CMSIS/DSP/Include -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/ruy -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/gemmlowp -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/kissfft -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/flatbuffers -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/cmsis/CMSIS/Core/Include -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/cmsis -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/flatbuffers/include -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/cmsis/CMSIS/NN/Include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_stdlib/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_gpio/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_base/include -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/generated/pico_base -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/boards/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_platform/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2040/hardware_regs/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_base/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2040/hardware_structs/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_claim/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_sync/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_uart/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_divider/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_time/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_timer/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_sync/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_util/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_runtime/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_clocks/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_irq/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_resets/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_pll/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_vreg/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_watchdog/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_xosc/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_printf/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_bootrom/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_bit_ops/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_divider/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_double/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_int64_ops/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_float/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_malloc/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/boot_stage2/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_binary_info/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_stdio/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_stdio_uart/include

C_FLAGS = -mcpu=cortex-m0plus -mthumb -O3 -DNDEBUG -nostdlib -ffunction-sections -fdata-sections -std=gnu11

CXX_DEFINES = -DCMSIS_NN=1 -DCOMPILE_DEFINITIONS -DLIB_PICO_BIT_OPS=1 -DLIB_PICO_BIT_OPS_PICO=1 -DLIB_PICO_DIVIDER=1 -DLIB_PICO_DIVIDER_HARDWARE=1 -DLIB_PICO_DOUBLE=1 -DLIB_PICO_DOUBLE_PICO=1 -DLIB_PICO_FLOAT=1 -DLIB_PICO_FLOAT_PICO=1 -DLIB_PICO_INT64_OPS=1 -DLIB_PICO_INT64_OPS_PICO=1 -DLIB_PICO_MALLOC=1 -DLIB_PICO_MEM_OPS=1 -DLIB_PICO_MEM_OPS_PICO=1 -DLIB_PICO_PLATFORM=1 -DLIB_PICO_PRINTF=1 -DLIB_PICO_PRINTF_PICO=1 -DLIB_PICO_RUNTIME=1 -DLIB_PICO_STANDARD_LINK=1 -DLIB_PICO_STDIO=1 -DLIB_PICO_STDIO_UART=1 -DLIB_PICO_STDLIB=1 -DLIB_PICO_SYNC=1 -DLIB_PICO_SYNC_CORE=1 -DLIB_PICO_SYNC_CRITICAL_SECTION=1 -DLIB_PICO_SYNC_MUTEX=1 -DLIB_PICO_SYNC_SEM=1 -DLIB_PICO_TIME=1 -DLIB_PICO_UTIL=1 -DPICO_BOARD=\"pico\" -DPICO_BUILD=1 -DPICO_CMAKE_BUILD_TYPE=\"Release\" -DPICO_COPY_TO_RAM=0 -DPICO_CXX_ENABLE_EXCEPTIONS=0 -DPICO_NO_FLASH=0 -DPICO_NO_HARDWARE=0 -DPICO_ON_DEVICE=1 -DPICO_TARGET_NAME=\"kernel_svdf_test\" -DPICO_USE_BLOCKED_RAM=0 -DTF_LITE_DISABLE_X86_NEON=1 -DTF_LITE_STATIC_MEMORY=1

CXX_INCLUDES = -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/tests/kernel_svdf_test/../../tests/kernel_svdf_test -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/cmsis/CMSIS/DSP/Include -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/ruy -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/gemmlowp -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/kissfft -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/flatbuffers -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/cmsis/CMSIS/Core/Include -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/cmsis -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/flatbuffers/include -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/pico-tflmicro/src/third_party/cmsis/CMSIS/NN/Include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_stdlib/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_gpio/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_base/include -I/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/generated/pico_base -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/boards/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_platform/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2040/hardware_regs/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_base/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2040/hardware_structs/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_claim/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_sync/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_uart/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_divider/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_time/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_timer/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_sync/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_util/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_runtime/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_clocks/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_irq/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_resets/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_pll/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_vreg/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_watchdog/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/hardware_xosc/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_printf/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_bootrom/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_bit_ops/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_divider/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_double/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_int64_ops/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_float/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_malloc/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/boot_stage2/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/common/pico_binary_info/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_stdio/include -I/home/sakib/working_dir/railcop/pico/pico-sdk/src/rp2_common/pico_stdio_uart/include

CXX_FLAGS = -mcpu=cortex-m0plus -mthumb -O3 -DNDEBUG -nostdlib -ffunction-sections -fdata-sections -fno-exceptions -fno-unwind-tables -fno-rtti -fno-use-cxa-atexit -std=gnu++11
