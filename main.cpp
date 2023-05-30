#include "whisper.cpp/whisper.h"
#include "whisper.cpp/ggml.h"

#define DR_WAV_IMPLEMENTATION
#include "whisper.cpp/examples/dr_wav.h"

#include <cmath>
#include <fstream>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <iostream>
#include "json/json.hpp"
#include <stdio.h>

using json = nlohmann::json;

typedef void (*progress_callback)(int progress);

char *jsonToChar(json jsonData) {
    std::string result = jsonData.dump(-1, ' ', false, json::error_handler_t::ignore);
    char *ch = new char[result.size() + 1];
    strcpy(ch, result.c_str());
    return ch;
}

// command-line parameters

struct whisper_params {
    int32_t n_threads    = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors =  1;
    int32_t offset_t_ms  =  0;
    int32_t offset_n     =  0;
    int32_t duration_ms  =  0;
    int32_t max_context  = -1;
    int32_t max_len      =  0;
    int32_t best_of      =  2;
    int32_t beam_size    = -1;

    float word_thold    =  0.01f;
    float entropy_thold =  2.40f;
    float logprob_thold = -1.00f;

    bool speed_up       = false;
    bool translate      = false;
    bool diarize        = false;
    bool split_on_word  = false;
    bool no_fallback    = false;
    bool output_txt     = false;
    bool output_vtt     = false;
    bool output_srt     = false;
    bool output_wts     = false;
    bool output_csv     = false;
    bool output_jsn     = false;
    bool output_lrc     = false;
    bool print_special  = false;
    bool print_colors   = false;
    bool print_progress = false;
    bool print_segments = false;
    bool no_timestamps  = false;

    std::string language = "en";
    std::string prompt;
    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model    = "models/ggml-base.en.bin";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};
};

bool is_aborted = false;

json transcribe(json jsonBody, progress_callback progress_cb) {
    whisper_params params;

    params.n_threads       = jsonBody["threads"];
    params.offset_t_ms     = jsonBody["offset_t_ms"];
    params.translate       = jsonBody["is_translate"];
    params.language        = jsonBody["language"];
    params.prompt          = jsonBody["prompt"];
    params.no_timestamps   = jsonBody["is_no_timestamps"];
    params.model           = jsonBody["model"];
    params.n_processors    = jsonBody["n_processors"];
    params.split_on_word   = jsonBody["split_on_word"];
    params.no_fallback     = jsonBody["no_fallback"];
    params.diarize         = jsonBody["diarize"];
    params.speed_up        = jsonBody["speed_up"];
    params.beam_size       = jsonBody["beam_size"];
    params.best_of         = jsonBody["best_of"];
    params.print_segments  = jsonBody["print_segments"];
    
    json jsonResult;
    jsonResult["@type"] = "transcribe";
    jsonResult["segments"] = {};

    // whisper init
    struct whisper_context *ctx = whisper_init_from_file(params.model.c_str());

    if (ctx == nullptr) {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "failed to initialize whisper context";
        return jsonResult;
    }

    std::vector<whisper_token> prompt_tokens;

    fprintf(stderr, "\n");
    fprintf(stderr, "initial prompt: '%s'\n", params.prompt.c_str());
    fprintf(stderr, "best_of: '%d'\n", params.best_of);
    fprintf(stderr, "beam_size: '%d'\n", params.beam_size);
    fprintf(stderr, "offset_t_ms: '%d'\n", params.offset_t_ms);

    std::string text_result = "";
    std::string fname_inp = jsonBody["audio"];
    // WAV input
    std::vector<float> pcmf32;
    {
        drwav wav;
        if (!drwav_init_file(&wav, fname_inp.c_str(), NULL)) {
            jsonResult["@type"] = "error";
            jsonResult["message"] = " failed to open WAV file ";
            return jsonResult;
        }

        fprintf(stderr, "input WAV: %d channels, %d Hz, %d frames\n", wav.channels, wav.sampleRate, (int)wav.totalPCMFrameCount);

        if (wav.channels != 1 && wav.channels != 2) {
            jsonResult["@type"] = "error";
            jsonResult["message"] = "must be mono or stereo";
            return jsonResult;
        }

        if (wav.sampleRate != WHISPER_SAMPLE_RATE) {
            jsonResult["@type"] = "error";
            jsonResult["message"] = "WAV file  must be 16 kHz";
            return jsonResult;
        }

        if (wav.bitsPerSample != 16) {
            jsonResult["@type"] = "error";
            jsonResult["message"] = "WAV file must be 16 bit";
            return jsonResult;
        }

        int n = wav.totalPCMFrameCount;

        std::vector<int16_t> pcm16;
        pcm16.resize(n * wav.channels);
        drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
        drwav_uninit(&wav);

        // convert to mono, float
        pcmf32.resize(n);
        if (wav.channels == 1) {
            for (uint64_t i = 0; i < n; i++) {
                pcmf32[i] = float(pcm16[i])/32768.0f;
            }
        } else {
            for (uint64_t i = 0; i < n; i++) {
                pcmf32[i] = float(pcm16[2*i] + pcm16[2*i + 1])/65536.0f;
            }
        }
    }

    // print some info about the processing
    {
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                printf("%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
    }

    // run the inference
    {
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

        wparams.print_realtime   = false;
        wparams.print_progress   = params.print_progress;
        wparams.print_timestamps = !params.no_timestamps;
        wparams.print_special    = params.print_special;
        wparams.translate        = params.translate;
        wparams.language         = params.language.c_str();
        wparams.n_threads        = params.n_threads;
        wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
        wparams.offset_ms        = params.offset_t_ms;
        wparams.duration_ms      = params.duration_ms;

        wparams.token_timestamps = params.output_wts || params.max_len > 0;
        wparams.thold_pt         = params.word_thold;
        wparams.max_len          = params.output_wts && params.max_len == 0 ? 60 : params.max_len;
        wparams.split_on_word    = params.split_on_word;

        wparams.speed_up         = params.speed_up;

        wparams.initial_prompt   = params.prompt.c_str();

        wparams.greedy.best_of        = params.best_of;
        wparams.beam_search.beam_size = params.beam_size;

        wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;
        wparams.entropy_thold    = params.entropy_thold;
        wparams.logprob_thold    = params.logprob_thold;


        wparams.progress_callback = [](struct whisper_context * ctx, struct whisper_state * state, int progress, void * user_data) {
            if (user_data) {
                progress_callback cb = (progress_callback)user_data;
                cb(progress);
            }
        };
        wparams.progress_callback_user_data = (void *)progress_cb;

        wparams.encoder_begin_callback = [](struct whisper_context * ctx, struct whisper_state * state, void * user_data) {
            bool* abort = (bool*)user_data;
            if (*abort) {
                *abort = false;
                return false;
            }
            return true;
        };
        wparams.encoder_begin_callback_user_data = &is_aborted;

        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
            jsonResult["@type"] = "error";
            jsonResult["message"] = "failed to process audio";
            return jsonResult;
        }

        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const char *text = whisper_full_get_segment_text(ctx, i);
            std::string str(text);

            if (params.print_segments) {
                printf("%s\n", text);
                fflush(stdout);
            }

            const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
            const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

            json segment;
            segment["text"] = str;
            segment["end"] = t1;
            segment["start"] = t0;
            jsonResult["segments"].push_back(segment);
        }
    }
    whisper_free(ctx);
    return jsonResult;
}

extern "C" {
    void stop_transcribe() {
        is_aborted = true;
    }

    char *request(char *body, progress_callback progress_cb) {
        json jsonBody = json::parse(body);
        json jsonResult;

        if (jsonBody["@type"] == "getTextFromWavFile") {
            return jsonToChar(transcribe(jsonBody, progress_cb));
        }

        if (jsonBody["@type"] == "getVersion") {
            jsonResult["@type"] = "version";
            jsonResult["message"] = "version lib v0.0.0";
            return jsonToChar(jsonResult);
        }

        jsonResult["@type"] = "error";
        jsonResult["message"] = "method not found";

        return jsonToChar(jsonResult);
    }
}

int main(int argc, char ** argv) {
    json jsonBody = json::parse(R"({
        "@type": "getTextFromWavFile",
        "model": "/Users/lei/Projects/audio-podium/macos/Runner/ggml-large.bin",
        "audio": "/Users/lei/Projects/hello-dart-ffi/space.wav",
        "threads": 4,
        "beam_size": 2,
        "best_of": 2,
        "offset_t_ms": 600000,
        "is_verbose": true,
        "is_translate": false,
        "prompt": "添加标点符号。",
        "language": "zh",
        "is_special_tokens": false,
        "is_no_timestamps": false,
        "n_processors": 1,
        "split_on_word": false,
        "no_fallback": false,
        "print_segments": true,
        "diarize": false,
        "speed_up": false
    })");
    json ret = transcribe(jsonBody, nullptr);
    printf("%s", jsonToChar(ret));
    return 0;
}