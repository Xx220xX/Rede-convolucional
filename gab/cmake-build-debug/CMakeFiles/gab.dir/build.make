# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.17

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

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2020.3.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2020.3.1\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\hslhe\Desktop\gab

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\hslhe\Desktop\gab\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/gab.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gab.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gab.dir/flags.make

CMakeFiles/gab.dir/gabriela_gpu.c.obj: CMakeFiles/gab.dir/flags.make
CMakeFiles/gab.dir/gabriela_gpu.c.obj: ../gabriela_gpu.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\hslhe\Desktop\gab\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/gab.dir/gabriela_gpu.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\gab.dir\gabriela_gpu.c.obj   -c C:\Users\hslhe\Desktop\gab\gabriela_gpu.c

CMakeFiles/gab.dir/gabriela_gpu.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/gab.dir/gabriela_gpu.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\hslhe\Desktop\gab\gabriela_gpu.c > CMakeFiles\gab.dir\gabriela_gpu.c.i

CMakeFiles/gab.dir/gabriela_gpu.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/gab.dir/gabriela_gpu.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\hslhe\Desktop\gab\gabriela_gpu.c -o CMakeFiles\gab.dir\gabriela_gpu.c.s

CMakeFiles/gab.dir/src/WrapperCL.c.obj: CMakeFiles/gab.dir/flags.make
CMakeFiles/gab.dir/src/WrapperCL.c.obj: ../src/WrapperCL.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\hslhe\Desktop\gab\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/gab.dir/src/WrapperCL.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\gab.dir\src\WrapperCL.c.obj   -c C:\Users\hslhe\Desktop\gab\src\WrapperCL.c

CMakeFiles/gab.dir/src/WrapperCL.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/gab.dir/src/WrapperCL.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\hslhe\Desktop\gab\src\WrapperCL.c > CMakeFiles\gab.dir\src\WrapperCL.c.i

CMakeFiles/gab.dir/src/WrapperCL.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/gab.dir/src/WrapperCL.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\hslhe\Desktop\gab\src\WrapperCL.c -o CMakeFiles\gab.dir\src\WrapperCL.c.s

CMakeFiles/gab.dir/src/Kernel.c.obj: CMakeFiles/gab.dir/flags.make
CMakeFiles/gab.dir/src/Kernel.c.obj: ../src/Kernel.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\hslhe\Desktop\gab\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/gab.dir/src/Kernel.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\gab.dir\src\Kernel.c.obj   -c C:\Users\hslhe\Desktop\gab\src\Kernel.c

CMakeFiles/gab.dir/src/Kernel.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/gab.dir/src/Kernel.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\hslhe\Desktop\gab\src\Kernel.c > CMakeFiles\gab.dir\src\Kernel.c.i

CMakeFiles/gab.dir/src/Kernel.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/gab.dir/src/Kernel.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\hslhe\Desktop\gab\src\Kernel.c -o CMakeFiles\gab.dir\src\Kernel.c.s

CMakeFiles/gab.dir/src/matGPU.c.obj: CMakeFiles/gab.dir/flags.make
CMakeFiles/gab.dir/src/matGPU.c.obj: ../src/matGPU.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\hslhe\Desktop\gab\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/gab.dir/src/matGPU.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\gab.dir\src\matGPU.c.obj   -c C:\Users\hslhe\Desktop\gab\src\matGPU.c

CMakeFiles/gab.dir/src/matGPU.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/gab.dir/src/matGPU.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\hslhe\Desktop\gab\src\matGPU.c > CMakeFiles\gab.dir\src\matGPU.c.i

CMakeFiles/gab.dir/src/matGPU.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/gab.dir/src/matGPU.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\hslhe\Desktop\gab\src\matGPU.c -o CMakeFiles\gab.dir\src\matGPU.c.s

CMakeFiles/gab.dir/src/lcg.c.obj: CMakeFiles/gab.dir/flags.make
CMakeFiles/gab.dir/src/lcg.c.obj: ../src/lcg.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\hslhe\Desktop\gab\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/gab.dir/src/lcg.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\gab.dir\src\lcg.c.obj   -c C:\Users\hslhe\Desktop\gab\src\lcg.c

CMakeFiles/gab.dir/src/lcg.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/gab.dir/src/lcg.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\hslhe\Desktop\gab\src\lcg.c > CMakeFiles\gab.dir\src\lcg.c.i

CMakeFiles/gab.dir/src/lcg.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/gab.dir/src/lcg.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\hslhe\Desktop\gab\src\lcg.c -o CMakeFiles\gab.dir\src\lcg.c.s

CMakeFiles/gab.dir/src/gabrielaLayer.c.obj: CMakeFiles/gab.dir/flags.make
CMakeFiles/gab.dir/src/gabrielaLayer.c.obj: ../src/gabrielaLayer.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\hslhe\Desktop\gab\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/gab.dir/src/gabrielaLayer.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\gab.dir\src\gabrielaLayer.c.obj   -c C:\Users\hslhe\Desktop\gab\src\gabrielaLayer.c

CMakeFiles/gab.dir/src/gabrielaLayer.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/gab.dir/src/gabrielaLayer.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\hslhe\Desktop\gab\src\gabrielaLayer.c > CMakeFiles\gab.dir\src\gabrielaLayer.c.i

CMakeFiles/gab.dir/src/gabrielaLayer.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/gab.dir/src/gabrielaLayer.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\hslhe\Desktop\gab\src\gabrielaLayer.c -o CMakeFiles\gab.dir\src\gabrielaLayer.c.s

# Object files for target gab
gab_OBJECTS = \
"CMakeFiles/gab.dir/gabriela_gpu.c.obj" \
"CMakeFiles/gab.dir/src/WrapperCL.c.obj" \
"CMakeFiles/gab.dir/src/Kernel.c.obj" \
"CMakeFiles/gab.dir/src/matGPU.c.obj" \
"CMakeFiles/gab.dir/src/lcg.c.obj" \
"CMakeFiles/gab.dir/src/gabrielaLayer.c.obj"

# External object files for target gab
gab_EXTERNAL_OBJECTS =

libgab.dll: CMakeFiles/gab.dir/gabriela_gpu.c.obj
libgab.dll: CMakeFiles/gab.dir/src/WrapperCL.c.obj
libgab.dll: CMakeFiles/gab.dir/src/Kernel.c.obj
libgab.dll: CMakeFiles/gab.dir/src/matGPU.c.obj
libgab.dll: CMakeFiles/gab.dir/src/lcg.c.obj
libgab.dll: CMakeFiles/gab.dir/src/gabrielaLayer.c.obj
libgab.dll: CMakeFiles/gab.dir/build.make
libgab.dll: CMakeFiles/gab.dir/linklibs.rsp
libgab.dll: CMakeFiles/gab.dir/objects1.rsp
libgab.dll: CMakeFiles/gab.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\hslhe\Desktop\gab\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking C shared library libgab.dll"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\gab.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gab.dir/build: libgab.dll

.PHONY : CMakeFiles/gab.dir/build

CMakeFiles/gab.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\gab.dir\cmake_clean.cmake
.PHONY : CMakeFiles/gab.dir/clean

CMakeFiles/gab.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\hslhe\Desktop\gab C:\Users\hslhe\Desktop\gab C:\Users\hslhe\Desktop\gab\cmake-build-debug C:\Users\hslhe\Desktop\gab\cmake-build-debug C:\Users\hslhe\Desktop\gab\cmake-build-debug\CMakeFiles\gab.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gab.dir/depend

