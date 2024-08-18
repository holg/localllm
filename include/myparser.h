// Copyright 2024 hjiay
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     https://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "localllm/llama.cpp/include/llama.h"
#include "rust/cxx.h"
#include <memory>

struct SamplerOption;
struct ModelOption;

LLAMA_API void load_model(const std::string &modelpath,bool initbackend,llama_model** model,llama_context** context,bool debug);
LLAMA_API void free_model(llama_model * model,llama_context* ctx,bool freebackend);
LLAMA_API std::unique_ptr<std::string> generate(llama_model * model,llama_context* ctx,const std::string &prompt,const rust::Slice<const std::array<rust::Str,2>>  history,struct llama_sampling_context * ctx_sampling,bool debug,ModelOption const & opt); 
LLAMA_API void init_sampler(struct llama_sampling_context ** ctx_sampling,const std::string &grammar_text,bool debug,SamplerOption const & config);
LLAMA_API void free_sampler(struct llama_sampling_context * ctx_sampling);
LLAMA_API void reset_sampler(llama_sampling_context * ctx);
LLAMA_API std::unique_ptr<std::string> json_schema_to_gbnf(const std::string& src);