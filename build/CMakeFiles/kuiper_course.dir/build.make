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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/horse/code/Kuiper/course2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/horse/code/Kuiper/course2/build

# Include any dependencies generated for this target.
include CMakeFiles/kuiper_course.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/kuiper_course.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/kuiper_course.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kuiper_course.dir/flags.make

CMakeFiles/kuiper_course.dir/main.cpp.o: CMakeFiles/kuiper_course.dir/flags.make
CMakeFiles/kuiper_course.dir/main.cpp.o: ../main.cpp
CMakeFiles/kuiper_course.dir/main.cpp.o: CMakeFiles/kuiper_course.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/horse/code/Kuiper/course2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/kuiper_course.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kuiper_course.dir/main.cpp.o -MF CMakeFiles/kuiper_course.dir/main.cpp.o.d -o CMakeFiles/kuiper_course.dir/main.cpp.o -c /home/horse/code/Kuiper/course2/main.cpp

CMakeFiles/kuiper_course.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kuiper_course.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/horse/code/Kuiper/course2/main.cpp > CMakeFiles/kuiper_course.dir/main.cpp.i

CMakeFiles/kuiper_course.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kuiper_course.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/horse/code/Kuiper/course2/main.cpp -o CMakeFiles/kuiper_course.dir/main.cpp.s

# Object files for target kuiper_course
kuiper_course_OBJECTS = \
"CMakeFiles/kuiper_course.dir/main.cpp.o"

# External object files for target kuiper_course
kuiper_course_EXTERNAL_OBJECTS =

../bin/kuiper_course: CMakeFiles/kuiper_course.dir/main.cpp.o
../bin/kuiper_course: CMakeFiles/kuiper_course.dir/build.make
../bin/kuiper_course: CMakeFiles/kuiper_course.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/horse/code/Kuiper/course2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/kuiper_course"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kuiper_course.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kuiper_course.dir/build: ../bin/kuiper_course
.PHONY : CMakeFiles/kuiper_course.dir/build

CMakeFiles/kuiper_course.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kuiper_course.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kuiper_course.dir/clean

CMakeFiles/kuiper_course.dir/depend:
	cd /home/horse/code/Kuiper/course2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/horse/code/Kuiper/course2 /home/horse/code/Kuiper/course2 /home/horse/code/Kuiper/course2/build /home/horse/code/Kuiper/course2/build /home/horse/code/Kuiper/course2/build/CMakeFiles/kuiper_course.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kuiper_course.dir/depend

