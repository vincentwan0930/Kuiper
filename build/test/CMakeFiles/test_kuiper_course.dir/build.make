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
CMAKE_SOURCE_DIR = /home/horse/code/KuiperCourse

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/horse/code/KuiperCourse/build

# Include any dependencies generated for this target.
include test/CMakeFiles/test_kuiper_course.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/test_kuiper_course.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_kuiper_course.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_kuiper_course.dir/flags.make

test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.o: test/CMakeFiles/test_kuiper_course.dir/flags.make
test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.o: ../test/test_first.cpp
test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.o: test/CMakeFiles/test_kuiper_course.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/horse/code/KuiperCourse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.o"
	cd /home/horse/code/KuiperCourse/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.o -MF CMakeFiles/test_kuiper_course.dir/test_first.cpp.o.d -o CMakeFiles/test_kuiper_course.dir/test_first.cpp.o -c /home/horse/code/KuiperCourse/test/test_first.cpp

test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_kuiper_course.dir/test_first.cpp.i"
	cd /home/horse/code/KuiperCourse/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/horse/code/KuiperCourse/test/test_first.cpp > CMakeFiles/test_kuiper_course.dir/test_first.cpp.i

test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_kuiper_course.dir/test_first.cpp.s"
	cd /home/horse/code/KuiperCourse/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/horse/code/KuiperCourse/test/test_first.cpp -o CMakeFiles/test_kuiper_course.dir/test_first.cpp.s

test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.o: test/CMakeFiles/test_kuiper_course.dir/flags.make
test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.o: ../test/test_main.cpp
test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.o: test/CMakeFiles/test_kuiper_course.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/horse/code/KuiperCourse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.o"
	cd /home/horse/code/KuiperCourse/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.o -MF CMakeFiles/test_kuiper_course.dir/test_main.cpp.o.d -o CMakeFiles/test_kuiper_course.dir/test_main.cpp.o -c /home/horse/code/KuiperCourse/test/test_main.cpp

test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_kuiper_course.dir/test_main.cpp.i"
	cd /home/horse/code/KuiperCourse/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/horse/code/KuiperCourse/test/test_main.cpp > CMakeFiles/test_kuiper_course.dir/test_main.cpp.i

test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_kuiper_course.dir/test_main.cpp.s"
	cd /home/horse/code/KuiperCourse/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/horse/code/KuiperCourse/test/test_main.cpp -o CMakeFiles/test_kuiper_course.dir/test_main.cpp.s

# Object files for target test_kuiper_course
test_kuiper_course_OBJECTS = \
"CMakeFiles/test_kuiper_course.dir/test_first.cpp.o" \
"CMakeFiles/test_kuiper_course.dir/test_main.cpp.o"

# External object files for target test_kuiper_course
test_kuiper_course_EXTERNAL_OBJECTS =

test/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.o
test/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.o
test/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/build.make
test/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/horse/code/KuiperCourse/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable test_kuiper_course"
	cd /home/horse/code/KuiperCourse/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_kuiper_course.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_kuiper_course.dir/build: test/test_kuiper_course
.PHONY : test/CMakeFiles/test_kuiper_course.dir/build

test/CMakeFiles/test_kuiper_course.dir/clean:
	cd /home/horse/code/KuiperCourse/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test_kuiper_course.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_kuiper_course.dir/clean

test/CMakeFiles/test_kuiper_course.dir/depend:
	cd /home/horse/code/KuiperCourse/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/horse/code/KuiperCourse /home/horse/code/KuiperCourse/test /home/horse/code/KuiperCourse/build /home/horse/code/KuiperCourse/build/test /home/horse/code/KuiperCourse/build/test/CMakeFiles/test_kuiper_course.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_kuiper_course.dir/depend

