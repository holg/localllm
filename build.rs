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

fn main() {
    cxx_build::bridge("src/llm.rs")
        .include("llama.cpp/ggml/include")
        .include("llama.cpp/include")
        .include("include")
        .file("src/main.cpp")
        .std("c++17")
        .compile("localllm");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/parse.cpp");
    println!("cargo:rerun-if-changed=include/myparser.h");
    println!("cargo:rustc-link-lib=llama");
    println!("cargo:rustc-link-lib=ggml");
    println!("cargo:rustc-link-lib=common");
    if let Ok(llama_lib_path) = std::env::var("LLAMA_LIB") {
        println!("cargo:rustc-link-search={llama_lib_path}");
    }

}