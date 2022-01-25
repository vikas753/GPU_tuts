# GPU_tuts

Upgrade to Windows 11 for WSL drivers support for GPU
1 . https://docs.nvidia.com/cuda/wsl-user-guide/index.html

cd /usr/local/cuda/samples
2 . < Start working on code > 

Local working directory : /usr/local/cuda/samples/8_custom_vikas/GPU_tuts

CUDA-GDB : /usr/local/cuda/bin

Exercise 3 Bugs Problem 3: 

1 . Cuda device synch was missing , for which a kernel error was wrongly being spit out as a memcpy error
2 . CUDA malloc and accessing it out of bounds of allocated memory region from kernel has no effect , unless memory access itself is completely outside of accessibility region of thread memory 