## Reproduction steps

1. Download the model file in the root dir of current repo.

```
https://huggingface.co/farmer00317558/quantized_whisper_model/resolve/main/ggml-medium-q5_0.bin
```

2. Pull git submodule

```bash
git submodule update --init --recursive
```

3. Build dylib

```bash
mkdir build
cd build
cmake ..
make
```

4. Run `bin/hello_dart_ffi.dart`

```bash
dart bin/hello_dart_ffi.dart
```

Output:

```
....

task complete in: 143s
```

5. Build C++ executable binary

```bash
cd whisper.cpp

```
