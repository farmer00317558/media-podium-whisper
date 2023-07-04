## Reproduction steps

1. Pull git submodule

```bash
git submodule update --init --recursive
cd whisper.cpp
git checkout v1.4.1
```

2. Build dylib

```bash
mkdir build
cd build
cmake ..
make
```

3. Run `bin/hello_dart_ffi.dart`

```bash
dart pub get
dart bin/hello_dart_ffi.dart
```

4. Build C++ executable binary

```bash
cd whisper.cpp
make
```

5. Run executable binary in dir `whisper.cpp`:

```bash
./main -m ../ggml-tiny-q5_0.bin -l zh -pp -bo 5 ../demo.wav
```