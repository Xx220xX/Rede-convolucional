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
CMAKE_SOURCE_DIR = C:\Users\hslhe\Desktop\cnn\CNN_GPU

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\hslhe\Desktop\cnn\CNN_GPU\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Library.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Library.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Library.dir/flags.make

CMakeFiles/Library.dir/CNNGPU.c.obj: CMakeFiles/Library.dir/flags.make
CMakeFiles/Library.dir/CNNGPU.c.obj: ../CNNGPU.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\hslhe\Desktop\cnn\CNN_GPU\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/Library.dir/CNNGPU.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\Library.dir\CNNGPU.c.obj   -c C:\Users\hslhe\Desktop\cnn\CNN_GPU\CNNGPU.c

CMakeFiles/Library.dir/CNNGPU.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Library.dir/CNNGPU.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\hslhe\Desktop\cnn\CNN_GPU\CNNGPU.c > CMakeFiles\Library.dir\CNNGPU.c.i

CMakeFiles/Library.dir/CNNGPU.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Library.dir/CNNGPU.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\hslhe\Desktop\cnn\CNN_GPU\CNNGPU.c -o CMakeFiles\Library.dir\CNNGPU.c.s

CMakeFiles/Library.dir/src/gpu/Kernel.c.obj: CMakeFiles/Library.dir/flags.make
CMakeFiles/Library.dir/src/gpu/Kernel.c.obj: ../src/gpu/Kernel.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\hslhe\Desktop\cnn\CNN_GPU\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/Library.dir/src/gpu/Kernel.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\Library.dir\src\gpu\Kernel.c.obj   -c C:\Users\hslhe\Desktop\cnn\CNN_GPU\src\gpu\Kernel.c

CMakeFiles/Library.dir/src/gpu/Kernel.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Library.dir/src/gpu/Kernel.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\hslhe\Desktop\cnn\CNN_GPU\src\gpu\Kernel.c > CMakeFiles\Library.dir\src\gpu\Kernel.c.i

CMakeFiles/Library.dir/src/gpu/Kernel.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Library.dir/src/gpu/Kernel.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\hslhe\Desktop\cnn\CNN_GPU\src\gpu\Kernel.c -o CMakeFiles\Library.dir\src\gpu\Kernel.c.s

CMakeFiles/Library.dir/src/gpu/lcg.c.obj: CMakeFiles/Library.dir/flags.make
CMakeFiles/Library.dir/src/gpu/lcg.c.obj: ../src/gpu/lcg.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\hslhe\Desktop\cnn\CNN_GPU\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/Library.dir/src/gpu/lcg.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\Library.dir\src\gpu\lcg.c.obj   -c C:\Users\hslhe\Desktop\cnn\CNN_GPU\src\gpu\lcg.c

CMakeFiles/Library.dir/src/gpu/lcg.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Library.dir/src/gpu/lcg.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\hslhe\Desktop\cnn\CNN_GPU\src\gpu\lcg.c > CMakeFiles\Library.dir\src\gpu\lcg.c.i

CMakeFiles/Library.dir/src/gpu/lcg.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Library.dir/src/gpu/lcg.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\hslhe\Desktop\cnn\CNN_GPU\src\gpu\lcg.c -o CMakeFiles\Library.dir\src\gpu\lcg.c.s

CMakeFiles/Library.dir/src/gpu/matGPU.c.obj: CMakeFiles/Library.dir/flags.make
CMakeFiles/Library.dir/src/gpu/matGPU.c.obj: ../src/gpu/matGPU.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\hslhe\Desktop\cnn\CNN_GPU\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/Library.dir/src/gpu/matGPU.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\Library.dir\src\gpu\matGPU.c.obj   -c C:\Users\hslhe\Desktop\cnn\CNN_GPU\src\gpu\matGPU.c

CMakeFiles/Library.dir/src/gpu/matGPU.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Library.dir/src/gpu/matGPU.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\hslhe\Desktop\cnn\CNN_GPU\src\gpu\matGPU.c > CMakeFiles\Library.dir\src\gpu\matGPU.c.i

CMakeFiles/Library.dir/src/gpu/matGPU.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Library.dir/src/gpu/matGPU.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\hslhe\Desktop\cnn\CNN_GPU\src\gpu\matGPU.c -o CMakeFiles\Library.dir\src\gpu\matGPU.c.s

CMakeFiles/Library.dir/src/gpu/WrapperCL.c.obj: CMakeFiles/Library.dir/flags.make
CMakeFiles/Library.dir/src/gpu/WrapperCL.c.obj: ../src/gpu/WrapperCL.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\hslhe\Desktop\cnn\CNN_GPU\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/Library.dir/src/gpu/WrapperCL.c.obj"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\Library.dir\src\gpu\WrapperCL.c.obj   -c C:\Users\hslhe\Desktop\cnn\CNN_GPU\src\gpu\WrapperCL.c

CMakeFiles/Library.dir/src/gpu/WrapperCL.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Library.dir/src/gpu/WrapperCL.c.i"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\hslhe\Desktop\cnn\CNN_GPU\src\gpu\WrapperCL.c > CMakeFiles\Library.dir\src\gpu\WrapperCL.c.i

CMakeFiles/Library.dir/src/gpu/WrapperCL.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Library.dir/src/gpu/WrapperCL.c.s"
	C:\MinGW\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\hslhe\Desktop\cnn\CNN_GPU\src\gpu\WrapperCL.c -o CMakeFiles\Library.dir\src\gpu\WrapperCL.c.s

# Object files for target Library
Library_OBJECTS = \
"CMakeFiles/Library.dir/CNNGPU.c.obj" \
"CMakeFiles/Library.dir/src/gpu/Kernel.c.obj" \
"CMakeFiles/Library.dir/src/gpu/lcg.c.obj" \
"CMakeFiles/Library.dir/src/gpu/matGPU.c.obj" \
"CMakeFiles/Library.dir/src/gpu/WrapperCL.c.obj"

# External object files for target Library
Library_EXTERNAL_OBJECTS =

libLibrary.dll: CMakeFiles/Library.dir/CNNGPU.c.obj
libLibrary.dll: CMakeFiles/Library.dir/src/gpu/Kernel.c.obj
libLibrary.dll: CMakeFiles/Library.dir/src/gpu/lcg.c.obj
libLibrary.dll: CMakeFiles/Library.dir/src/gpu/matGPU.c.obj
libLibrary.dll: CMakeFiles/Library.dir/src/gpu/WrapperCL.c.obj
libLibrary.dll: CMakeFiles/Library.dir/build.make
libLibrary.dll: CMakeFiles/Library.dir/linklibs.rsp
libLibrary.dll: CMakeFiles/Library.dir/objects1.rsp
libLibrary.dll: CMakeFiles/Library.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\hslhe\Desktop\cnn\CNN_GPU\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking C shared library libLibrary.dll"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Library.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Library.dir/build: libLibrary.dll

.PHONY : CMakeFiles/Library.dir/build

CMakeFiles/Library.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Library.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Library.dir/clean

CMakeFiles/Library.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\hslhe\Desktop\cnn\CNN_GPU C:\Users\hslhe\Desktop\cnn\CNN_GPU C:\Users\hslhe\Desktop\cnn\CNN_GPU\cmake-build-debug C:\Users\hslhe\Desktop\cnn\CNN_GPU\cmake-build-debug C:\Users\hslhe\Desktop\cnn\CNN_GPU\cmake-build-debug\CMakeFiles\Library.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Library.dir/depend
