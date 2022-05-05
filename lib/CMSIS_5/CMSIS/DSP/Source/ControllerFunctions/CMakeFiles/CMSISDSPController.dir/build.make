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
include lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/progress.make

# Include the compile flags for this target's objects.
include lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/flags.make

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_init_f32.c
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.obj -MF CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.obj.d -o CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_init_f32.c

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_init_f32.c > CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.i

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_init_f32.c -o CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.s

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_init_q15.c
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.obj -MF CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.obj.d -o CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_init_q15.c

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_init_q15.c > CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.i

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_init_q15.c -o CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.s

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_init_q31.c
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.obj -MF CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.obj.d -o CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_init_q31.c

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_init_q31.c > CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.i

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_init_q31.c -o CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.s

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_reset_f32.c
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.obj -MF CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.obj.d -o CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_reset_f32.c

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_reset_f32.c > CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.i

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_reset_f32.c -o CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.s

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_reset_q15.c
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.obj -MF CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.obj.d -o CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_reset_q15.c

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_reset_q15.c > CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.i

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_reset_q15.c -o CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.s

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_reset_q31.c
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.obj -MF CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.obj.d -o CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_reset_q31.c

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_reset_q31.c > CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.i

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_pid_reset_q31.c -o CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.s

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_sin_cos_f32.c
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.obj -MF CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.obj.d -o CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_sin_cos_f32.c

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_sin_cos_f32.c > CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.i

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_sin_cos_f32.c -o CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.s

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/flags.make
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_sin_cos_q31.c
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.obj: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.obj"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.obj -MF CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.obj.d -o CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.obj -c /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_sin_cos_q31.c

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.i"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_sin_cos_q31.c > CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.i

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.s"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && /usr/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/arm_sin_cos_q31.c -o CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.s

# Object files for target CMSISDSPController
CMSISDSPController_OBJECTS = \
"CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.obj" \
"CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.obj" \
"CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.obj" \
"CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.obj" \
"CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.obj" \
"CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.obj" \
"CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.obj" \
"CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.obj"

# External object files for target CMSISDSPController
CMSISDSPController_EXTERNAL_OBJECTS =

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/libCMSISDSPController.a: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_f32.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/libCMSISDSPController.a: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q15.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/libCMSISDSPController.a: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_init_q31.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/libCMSISDSPController.a: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_f32.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/libCMSISDSPController.a: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q15.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/libCMSISDSPController.a: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_pid_reset_q31.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/libCMSISDSPController.a: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_f32.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/libCMSISDSPController.a: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/arm_sin_cos_q31.c.obj
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/libCMSISDSPController.a: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/build.make
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/libCMSISDSPController.a: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking C static library libCMSISDSPController.a"
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && $(CMAKE_COMMAND) -P CMakeFiles/CMSISDSPController.dir/cmake_clean_target.cmake
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CMSISDSPController.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/build: lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/libCMSISDSPController.a
.PHONY : lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/build

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/clean:
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions && $(CMAKE_COMMAND) -P CMakeFiles/CMSISDSPController.dir/cmake_clean.cmake
.PHONY : lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/clean

lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/depend:
	cd /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions /home/sakib/working_dir/railcop/ml-audio-classifier-example-for-pico/inference-app/lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/CMSIS_5/CMSIS/DSP/Source/ControllerFunctions/CMakeFiles/CMSISDSPController.dir/depend

