### this is a **fork** from [**FLASHLIGHT**](https://github.com/flashlight/flashlight)



### State
[![Build & Test](https://github.com/codethinki/flashmini/actions/workflows/build-test.yml/badge.svg)](https://github.com/codethinki/flashmini/actions/workflows/build-test.yml)
Badge indicates:
- Build with gcc (linux) and msvc (windows) with both cpu & cuda backend
- Testing: gcc & msvc on cpu backend

The Tests should also run on cuda there is just no CI for it (github free runner limitation)

does not compile on (yet):
- clang, clang-cl (non standard behavior in forward decls, gotta fix that first)

compiles not (yet):
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
    - oneDNN (only standalone version)
    - JIT

### [Code of Conduct](CODE_OF_CONDUCT.md)
