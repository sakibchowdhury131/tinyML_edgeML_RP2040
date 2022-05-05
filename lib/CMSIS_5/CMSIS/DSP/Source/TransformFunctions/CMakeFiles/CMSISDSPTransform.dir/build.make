# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/sakib/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/sakib/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app

# Include any dependencies generated for this target.
include lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/progress.make

# Include the compile flags for this target's objects.
include lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/flags.make

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal.c
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.obj -MF CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.obj.d -o CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal.c

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal.c > CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.i

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal.c -o CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.s

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal2.c
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.obj -MF CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.obj.d -o CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal2.c

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal2.c > CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.i

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal2.c -o CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.s

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal_f16.c
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.obj -MF CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.obj.d -o CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal_f16.c

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal_f16.c > CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.i

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_bitreversal_f16.c -o CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.s

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_rfft_init_q15.c
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.obj -MF CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.obj.d -o CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_rfft_init_q15.c

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_rfft_init_q15.c > CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.i

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_rfft_init_q15.c -o CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.s

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_rfft_q15.c
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.obj -MF CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.obj.d -o CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_rfft_q15.c

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_rfft_q15.c > CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.i

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_rfft_q15.c -o CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.s

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_cfft_q15.c
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.obj -MF CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.obj.d -o CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_cfft_q15.c

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_cfft_q15.c > CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.i

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_cfft_q15.c -o CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.s

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_cfft_init_q15.c
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.obj -MF CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.obj.d -o CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_cfft_init_q15.c

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_cfft_init_q15.c > CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.i

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_cfft_init_q15.c -o CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.s

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_cfft_radix4_q15.c
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.obj -MF CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.obj.d -o CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_cfft_radix4_q15.c

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_cfft_radix4_q15.c > CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.i

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/arm_cfft_radix4_q15.c -o CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.s

# Object files for target CMSISDSPTransform
CMSISDSPTransform_OBJECTS = \
"CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.obj" \
"CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.obj" \
"CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.obj" \
"CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.obj" \
"CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.obj" \
"CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.obj" \
"CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.obj" \
"CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.obj"

# External object files for target CMSISDSPTransform
CMSISDSPTransform_EXTERNAL_OBJECTS =

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/libCMSISDSPTransform.a: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/libCMSISDSPTransform.a: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal2.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/libCMSISDSPTransform.a: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_bitreversal_f16.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/libCMSISDSPTransform.a: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_init_q15.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/libCMSISDSPTransform.a: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_rfft_q15.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/libCMSISDSPTransform.a: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_q15.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/libCMSISDSPTransform.a: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_init_q15.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/libCMSISDSPTransform.a: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/arm_cfft_radix4_q15.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/libCMSISDSPTransform.a: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/build.make
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/libCMSISDSPTransform.a: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking C static library libCMSISDSPTransform.a"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && $(CMAKE_COMMAND) -P CMakeFiles/CMSISDSPTransform.dir/cmake_clean_target.cmake
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CMSISDSPTransform.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/build: lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/libCMSISDSPTransform.a
.PHONY : lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/build

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/clean:
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions && $(CMAKE_COMMAND) -P CMakeFiles/CMSISDSPTransform.dir/cmake_clean.cmake
.PHONY : lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/clean

lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/depend:
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/CMSIS_5/CMSIS/DSP/Source/TransformFunctions/CMakeFiles/CMSISDSPTransform.dir/depend

