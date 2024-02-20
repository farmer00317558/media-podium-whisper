import 'dart:convert';
import 'dart:io' show Directory, Platform;
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:path/path.dart' as path;

typedef ProgressCallback = Void Function(Int32);

typedef RequestTranscribe = Pointer<Utf8> Function(
  Pointer<Utf8>,
  Pointer<NativeFunction<ProgressCallback>>,
);

var homePath = Platform.environment['HOME'] ?? "";
var libName = "libmedia_podium_whisper.dylib";

var libPath = path.join(Directory.current.path, 'build', libName);
// var wavFilePath = path.join(Directory.current.path, 'demo.wav');
var modelFilePath =
    path.join(homePath, 'Models/whisper/ggml-large-v3-q5_0.bin');
var wavFilePath =
    "/Users/lei/Projects/media-podium-whisper/whisper.cpp/samples/jfk.wav";

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

    final params = jsonEncode(
      {
        "@type": "transcribe",
        "model": modelFilePath,
        "file": wavFilePath,
        "language": "auto",
        "diarize": true,
        "use_gpu": true,
      },
    );

    print(params);
    final res = request(
      params.toNativeUtf8(),
      Pointer.fromFunction(Task.progress),
    );

    print('task complete in: ${DateTime.now().difference(start).inSeconds}s');
    final data = jsonDecode(res.toDartString()) as Map<String, dynamic>;
    for (var i in data["segments"] as List<dynamic>) {
      print("${i['start']} --> ${i['end']}: ${i['text']}");
    }
  }
}
