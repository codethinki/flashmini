### this is a **fork** from [**FLASHLIGHT**](https://github.com/flashlight/flashlight)



### State

compiles on:
- msvc windows
- gcc linux

does not compile on:
- clang, clang-cl (non standard behavior in forward decls, gotta fix that first)

compiles not:
- FL_USE_CUDNN bc of old api


### [Contributing](CONTRIBUTING.md)
Please read the [todo list](TODO.md)


### Quirks
`FL_USE_CUDNN`:
- NOT WORKING ATM (v6-7 api from 2017 i gotta fix that first)
- requires `CUDNN_ROOT` to be set in the environment
- windows users: **do not** install CUDNN with the default **windows installer**. It will create: `CUDNN_ROOT/<include/bin>/<cuda-version>/`. Since i cannot anticipate the cuda version you use, i can't traverse this. **FIX:** install as **tarball** instead.  

### Functional changes from Flashlight
- backends removed:
    - oneDNN
    - JIT

### [Code of Conduct](CODE_OF_CONDUCT.md)
