#include "whisper.cpp/examples/common.h"
#include "whisper.cpp/examples/common-whisper.h"
#include "whisper.cpp/include/whisper.h"
#include "whisper.cpp/examples/grammar-parser.h"


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

struct whisper_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t offset_n      = 0;
    int32_t duration_ms   = 0;
    int32_t progress_step = 5;
    int32_t max_context   = -1;
    int32_t max_len       = 0;
    int32_t best_of       = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size     = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t audio_ctx     = 0;

    float word_thold      =  0.01f;
    float entropy_thold   =  2.40f;
    float logprob_thold   = -1.00f;
    float no_speech_thold =  0.3f;
    float grammar_penalty = 100.0f;
    float temperature     = 0.0f;
    float temperature_inc = 0.2f;

    bool debug_mode      = true;
    bool translate       = false;
    bool detect_language = false;
    bool diarize         = false;
    bool tinydiarize     = false;
    bool split_on_word   = false;
    bool no_fallback     = false;
    bool print_special   = false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool log_score       = false;
    bool use_gpu         = true;
    bool flash_attn      = false;
    bool suppress_nst    = false;

    std::string language  = "en";
    std::string prompt;
    std::string model     = "models/ggml-base.en.bin";
    std::string grammar;
    std::string grammar_rule;

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]"; // TODO: set from command line

    // A regular expression that matches tokens to suppress
    std::string suppress_regex;

    std::string openvino_encode_device = "CPU";

    std::string dtw = "";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

    grammar_parser::parse_state grammar_parsed;
};

struct whisper_print_user_data {
    const whisper_params * params;
    const std::vector<std::vector<float>> * pcmf32s;
    int progress_prev;
};

char *jsonToChar(json jsonData) {
    std::string result = jsonData.dump(-1, ' ', false, json::error_handler_t::ignore);
    char *ch = new char[result.size() + 1];
    strcpy(ch, result.c_str());
    return ch;
}

static std::string estimate_diarization_speaker(std::vector<std::vector<float>> pcmf32s, int64_t t0, int64_t t1, bool id_only = false) {
    std::string speaker = "";
    const int64_t n_samples = pcmf32s[0].size();

    const int64_t is0 = timestamp_to_sample(t0, n_samples, WHISPER_SAMPLE_RATE);
    const int64_t is1 = timestamp_to_sample(t1, n_samples, WHISPER_SAMPLE_RATE);

    double energy0 = 0.0f;
    double energy1 = 0.0f;

    for (int64_t j = is0; j < is1; j++) {
        energy0 += fabs(pcmf32s[0][j]);
        energy1 += fabs(pcmf32s[1][j]);
    }

    if (energy0 > 1.1*energy1) {
        speaker = "0";
    } else if (energy1 > 1.1*energy0) {
        speaker = "1";
    } else {
        speaker = "?";
    }

    if (!id_only) {
        speaker.insert(0, "(speaker ");
        speaker.append(")");
    }

    return speaker;
}

static void whisper_print_progress_callback(struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void * user_data) {
    int progress_step = ((whisper_print_user_data *) user_data)->params->progress_step;
    int * progress_prev  = &(((whisper_print_user_data *) user_data)->progress_prev);
    if (progress >= *progress_prev + progress_step) {
        *progress_prev += progress_step;
        fprintf(stderr, "%s: progress = %3d%%\n", __func__, progress);
    }
}

static void whisper_print_segment_callback(struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data) {
    const auto & params  = *((whisper_print_user_data *) user_data)->params;
    const auto & pcmf32s = *((whisper_print_user_data *) user_data)->pcmf32s;

    const int n_segments = whisper_full_n_segments(ctx);

    std::string speaker = "";

    int64_t t0 = 0;
    int64_t t1 = 0;

    // print the last n_new segments
    const int s0 = n_segments - n_new;

    if (s0 == 0) {
        printf("\n");
    }

    for (int i = s0; i < n_segments; i++) {
        if (!params.no_timestamps || params.diarize) {
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }

        if (!params.no_timestamps) {
            printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
        }

        if (params.diarize && pcmf32s.size() == 2) {
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        const char * text = whisper_full_get_segment_text(ctx, i);
        printf("%s%s", speaker.c_str(), text);

        if (params.tinydiarize) {
            if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                printf("%s", params.tdrz_speaker_turn.c_str());
            }
        }

        // with timestamps or speakers: each segment on new line
        if (!params.no_timestamps || params.diarize) {
            printf("\n");
        }

        fflush(stdout);
    }
}

whisper_params whisper_params_parse(json data) {
    struct whisper_params params;
    if (data["threads"].is_number_integer()     ) { params.n_threads              = data["threads"].get<int32_t>() ;       }
    if (data["processors"].is_number_integer()  ) { params.n_processors           = data["processors"].get<int32_t>();     }
    if (data["offset-t"].is_number_integer()    ) { params.offset_t_ms            = data["offset-t"].get<int32_t>();       }
    if (data["offset-n"].is_number_integer()    ) { params.offset_n               = data["offset-n"].get<int32_t>();       }
    if (data["duration"].is_number_integer()    ) { params.duration_ms            = data["duration"].get<int32_t>();       }
    if (data["max-context"].is_number_integer() ) { params.max_context            = data["max-context"].get<int32_t>();    }
    if (data["max-len"].is_number_integer()     ) { params.max_len                = data["max-len"].get<int32_t>();        }
    if (data["best-of"].is_number_integer()     ) { params.best_of                = data["best-of"].get<int32_t>();        }
    if (data["beam-size"].is_number_integer()   ) { params.beam_size              = data["beam-size"].get<int32_t>();      }
    if (data["word-thold"].is_number_float()    ) { params.word_thold             = data["word-thold"].get<float>();       }
    if (data["entropy-thold"].is_number_float() ) { params.entropy_thold          = data["entropy-thold"].get<float>();    }
    if (data["logprob-thold"].is_number_float() ) { params.logprob_thold          = data["logprob-thold"].get<float>();    }
    if (data["language"].is_string()            ) { params.language               = data["language"].get<std::string>();   }
    if (data["prompt"].is_string()              ) { params.prompt                 = data["prompt"].get<std::string>();     }
    if (data["model"].is_string()               ) { params.model                  = data["model"].get<std::string>();      }
    if (data["translate"].is_boolean()          ) { params.translate              = data["translate"].get<bool>();         }
    if (data["use_gpu"].is_boolean()            ) { params.use_gpu                = data["use_gpu"].get<bool>();           }
    if (data["dtw"].is_string()                 ) { params.dtw                    = data["dtw"].get<std::string>();        }
    if (data["ov-e-device"].is_string()         ) { params.openvino_encode_device = data["ov-e-device"].get<std::string>();}
    if (data["file"].is_string()                ) { params.fname_inp.emplace_back(data["file"]);                           }
    params.debug_mode             = false; 
    params.diarize                = false; 
    params.tinydiarize            = false; 
    params.split_on_word          = false; 
    params.no_fallback            = false; 
    params.print_special          = false; 
    params.print_progress         = true; 
    params.no_timestamps          = false; 
    params.detect_language        = false;
    params.log_score              = false;
    return params;
}

std::atomic<bool> is_aborted(false);

json transcribe(json jsonBody, progress_callback progress_cb) {
    json jsonResult;
    jsonResult["@type"] = "transcribe";
    jsonResult["segments"] = {};

    whisper_params params = whisper_params_parse(jsonBody);

    if (params.fname_inp.empty()) {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "no input files specified";
        return jsonResult;
    }

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "unknown language";
        return jsonResult;
    }

    if (params.diarize && params.tinydiarize) {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "error: cannot use both --diarize and --tinydiarize";
        return jsonResult;
    }

    // whisper init
    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    if (!params.dtw.empty()) {
        cparams.dtw_token_timestamps = true;
        cparams.dtw_aheads_preset = WHISPER_AHEADS_NONE;

        // fprintf(stderr, "dtw: '%s'\n", params.dtw.c_str());

        if (params.dtw == "tiny")      cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY;
        if (params.dtw == "tiny.en")   cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY_EN;
        if (params.dtw == "base")      cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE;
        if (params.dtw == "base.en")   cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;
        if (params.dtw == "small")     cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL;
        if (params.dtw == "small.en")  cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL_EN;
        if (params.dtw == "medium")    cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM;
        if (params.dtw == "medium.en") cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM_EN;
        if (params.dtw == "large.v1")  cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V1;
        if (params.dtw == "large.v2")  cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V2;
        if (params.dtw == "large.v3")  cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3;
        if (params.dtw == "large.v3.turbo")  cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3_TURBO;

        if (cparams.dtw_aheads_preset == WHISPER_AHEADS_NONE) {
            fprintf(stderr, "error: unknown DTW preset '%s'\n", params.dtw.c_str());
            jsonResult["@type"] = "error";
            jsonResult["message"] = "error: unknown DTW preset";
            return jsonResult;
        }
    }

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr) {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "error: failed to initialize whisper context";
        return jsonResult;
    }

    // initialize openvino encoder. this has no effect on whisper.cpp builds that don't have OpenVINO configured
    whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr);

    if (!params.grammar.empty()) {
        auto & grammar = params.grammar_parsed;
        if (is_file_exist(params.grammar.c_str())) {
            // read grammar from file
            std::ifstream ifs(params.grammar.c_str());
            const std::string txt = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            grammar = grammar_parser::parse(txt.c_str());
        } else {
            // read grammar from string
            grammar = grammar_parser::parse(params.grammar.c_str());
        }

        // will be empty (default) if there are parse errors
        if (grammar.rules.empty()) {
            fprintf(stderr, "error: failed to parse grammar \"%s\"\n", params.grammar.c_str());
            jsonResult["@type"] = "error";
            jsonResult["message"] = "error: failed to parse grammar";
            return jsonResult;
        } else {
            fprintf(stderr, "%s: grammar:\n", __func__);
            grammar_parser::print_grammar(stderr, grammar);
            fprintf(stderr, "\n");
        }
    }
    // for (int f = 0; f < (int) params.fname_inp.size(); ++f) {
    for (int f = 0; f < 1; ++f) {
        const auto fname_inp = params.fname_inp[f];
        const auto fname_out = f < (int) params.fname_out.size() && !params.fname_out[f].empty() ? params.fname_out[f] : params.fname_inp[f];

        std::vector<float> pcmf32;               // mono-channel F32 PCM
        std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

        if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, params.diarize)) {
            fprintf(stderr, "error: failed to read audio file '%s'\n", fname_inp.c_str());
            jsonResult["@type"] = "error";
            jsonResult["message"] = "error: failed to read audio file";
            return jsonResult;
        }

        // print system information
        {
            fprintf(stderr, "\n");
            fprintf(stderr, "system_info: n_threads = %d / %d\n", params.n_threads*params.n_processors, std::thread::hardware_concurrency());
        }

        {
            fprintf(stderr, "\n");
            if (!whisper_is_multilingual(ctx)) {
                if (params.language != "en" || params.translate) {
                    params.language = "en";
                    params.translate = false;
                    fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
                }
            }
            if (params.detect_language) {
                params.language = "auto";
            }
            fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d processors, %d beams + best of %d, lang = %s, task = %s, %stimestamps = %d ...\n",
                    __func__, fname_inp.c_str(), int(pcmf32.size()), float(pcmf32.size())/WHISPER_SAMPLE_RATE,
                    params.n_threads, params.n_processors, params.beam_size, params.best_of,
                    params.language.c_str(),
                    params.translate ? "translate" : "transcribe",
                    params.tinydiarize ? "tdrz = 1, " : "",
                    params.no_timestamps ? 0 : 1);

            fprintf(stderr, "\n");
        }

        // run the inference
        {
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            const bool use_grammar = (!params.grammar_parsed.rules.empty() && !params.grammar_rule.empty());
            wparams.strategy = (params.beam_size > 1 || use_grammar) ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

            wparams.print_realtime   = false;
            wparams.print_progress   = params.print_progress;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.print_special    = params.print_special;
            wparams.translate        = params.translate;
            wparams.language         = params.language.c_str();
            wparams.detect_language  = params.detect_language;
            wparams.n_threads        = params.n_threads;
            wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
            wparams.offset_ms        = params.offset_t_ms;
            wparams.duration_ms      = params.duration_ms;

            wparams.token_timestamps = params.max_len > 0;
            wparams.thold_pt         = params.word_thold;
            wparams.max_len          = params.max_len == 0 ? 60 : params.max_len;
            wparams.split_on_word    = params.split_on_word;
            wparams.audio_ctx        = params.audio_ctx;

            wparams.debug_mode       = params.debug_mode;

            wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

            wparams.suppress_regex   = params.suppress_regex.empty() ? nullptr : params.suppress_regex.c_str();

            wparams.initial_prompt   = params.prompt.c_str();

            wparams.greedy.best_of        = params.best_of;
            wparams.beam_search.beam_size = params.beam_size;

            wparams.temperature_inc  = params.no_fallback ? 0.0f : params.temperature_inc;
            wparams.temperature      = params.temperature;

            wparams.entropy_thold    = params.entropy_thold;
            wparams.logprob_thold    = params.logprob_thold;
            wparams.no_speech_thold  = params.no_speech_thold;

            wparams.no_timestamps    = params.no_timestamps;

            wparams.suppress_nst     = params.suppress_nst;

            whisper_print_user_data user_data = { &params, &pcmf32s, 0 };

            const auto & grammar_parsed = params.grammar_parsed;
            auto grammar_rules = grammar_parsed.c_rules();

            if (use_grammar) {
                if (grammar_parsed.symbol_ids.find(params.grammar_rule) == grammar_parsed.symbol_ids.end()) {
                    fprintf(stderr, "%s: warning: grammar rule '%s' not found - skipping grammar sampling\n", __func__, params.grammar_rule.c_str());
                } else {
                    wparams.grammar_rules = grammar_rules.data();
                    wparams.n_grammar_rules = grammar_rules.size();
                    wparams.i_start_rule = grammar_parsed.symbol_ids.at(params.grammar_rule);
                    wparams.grammar_penalty = params.grammar_penalty;
                }
            }

            // this callback is called on each new segment
            if (!wparams.print_realtime) {
                wparams.new_segment_callback           = whisper_print_segment_callback;
                wparams.new_segment_callback_user_data = &user_data;
            }

            if (wparams.print_progress) {
                wparams.progress_callback           = whisper_print_progress_callback;
                wparams.progress_callback_user_data = &user_data;
            }

            // examples for abort mechanism
            // in examples below, we do not abort the processing, but we could if the flag is set to true

            // the callback is called before every encoder run - if it returns false, the processing is aborted
            {
                wparams.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
                    auto is_aborted = static_cast<std::atomic<bool>*>(user_data);
                    return !is_aborted->load();
                };

                wparams.encoder_begin_callback_user_data = &is_aborted;
            }

            // the callback is called before every computation - if it returns true, the computation is aborted
            {
                wparams.abort_callback = [](void * user_data) {
                    auto is_aborted = static_cast<std::atomic<bool>*>(user_data);
                    return is_aborted->load();
                };
                wparams.abort_callback_user_data = &is_aborted;
            }

            if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
                fprintf(stderr, "failed to process audio\n");
                jsonResult["@type"] = "error";
                jsonResult["message"] = "inference failed";
                return jsonResult;
            }

            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char * text = whisper_full_get_segment_text(ctx, i);
                const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                std::string speaker = "";

                if (params.diarize && pcmf32s.size() == 2)
                {
                    speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
                }

                json segment;
                segment["text"] = text;
                segment["end"] = t1;
                segment["start"] = t0;
                segment["speaker"] = speaker;
                jsonResult["segments"].push_back(segment);
            }
        }
    }
    whisper_free(ctx);
    return jsonResult;
}

void abort_transcription(std::atomic<bool>& is_aborted) {
    is_aborted.store(true);
}

extern "C" {
    void stop_transcribe() {
        abort_transcription(is_aborted);
    }

    char *request(char *body, progress_callback progress_cb) {
        json jsonBody = json::parse(body);
        json jsonResult;

        if (jsonBody["@type"] == "transcribe") {
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
        "@type": "transcribe",
        "model": "/Users/lei/Models/whisper/ggml-large-v3-q5_0.bin",
        "file": "/Users/lei/Lab/whisper.cpp/kedaibiaolizheng1.wav",
        "language": "auto"
    })");

    std::thread transcription_thread([&]() {
        transcribe(jsonBody, nullptr);
    });


    std::this_thread::sleep_for(std::chrono::seconds(10));
    abort_transcription(is_aborted);
    
    transcription_thread.join();
}