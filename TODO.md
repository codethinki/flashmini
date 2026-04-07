# TODOS

## Roadmap

- fix cmake
	- fix frankenstein: Currently the whole thing is just one target with conditionally added sources, i'd like to split that into multiple targets conditionally composed
	- add assertions instead of ifs (using the cmake utility stuff)
  
- fix build
	- clang build: the dependencies in this code between source files are non standard (partially), refactoring is necessary


## Dependencies

### vcpkg
- add arrayfire (currently not possible bc the vcpkg port crashes the configure for newer cuda versions)
