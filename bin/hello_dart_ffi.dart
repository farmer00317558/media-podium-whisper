import 'dart:convert';
import 'dart:io' show Platform, Directory;
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:path/path.dart' as path;

typedef ProgressCallback = Void Function(Int32);

typedef RequestTranscribe = Pointer<Utf8> Function(
  Pointer<Utf8>,
  Pointer<NativeFunction<ProgressCallback>>,
);

var libPath = path.join(Directory.current.path, 'build', 'libwhisper.dylib');
var wavFilePath = path.join(Directory.current.path, 'demo.wav');
var modelFilePath = path.join(Directory.current.path, 'ggml-tiny-q5_0.bin');

void main(List<String> arguments) {
  Task.run();
}

class Task {
  static void progress(int i) {
    print('progress: $i%');
  }

  static void run() {
    print('start task...');
    final start = DateTime.now();
    final dylib = DynamicLibrary.open(libPath);
    final request =
        dylib.lookupFunction<RequestTranscribe, RequestTranscribe>('request');

    final res = request(
      jsonEncode(
        {
          "@type": "getTextFromWavFile",
          "model": modelFilePath,
          "audio": wavFilePath,
          "threads": 4,
          "is_verbose": true,
          "is_translate": false,
          "prompt": "",
          "language": "zh",
          "is_special_tokens": false,
          "is_no_timestamps": false,
          "n_processors": 1,
          "split_on_word": false,
          "no_fallback": false,
          "diarize": false,
          "speed_up": false,
        },
      ).toNativeUtf8(),
      Pointer.fromFunction(Task.progress),
    );

    print('task complete in: ${DateTime.now().difference(start).inSeconds}s');
  }
}
