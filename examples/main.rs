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

use localllm::{Model, Sampler, Tool, ToolPrompt};

use localllm::schema_for;
use localllm::Deserialize;
use localllm::JsonSchema;
#[derive(JsonSchema, Deserialize)]
struct Add {
    #[schemars(description = "Addition Left Hand Side")]
    #[schemars(regex(pattern = r"^[0-9]+$"))]
    pub lhs: String,
    #[schemars(description = "Addition Right Hand Side")]
    pub rhs: i32,
}

#[derive(JsonSchema, Deserialize)]
struct Sub {
    #[schemars(description = "Subtraction Left Hand Side.")]
    pub lhs: i32,
    #[schemars(description = "Subtraction Right Hand Side.")]
    pub rhs: i32,
}

fn main() {
    let schema_add: schemars::schema::RootSchema = schema_for!(Add);
    let schema_sub: schemars::schema::RootSchema = schema_for!(Sub);

    let schemastring = serde_json::to_string(&schema_add.schema).unwrap();
    let gbnfstring = localllm::json_schema_to_gbnf(&schemastring);
    println!("gbnf:\n{}", gbnfstring);
    let tokens = localllm::tokens(&gbnfstring);
    println!("tokens:\n{:?}", tokens);
    println!(
        "{}, {}",
        gbnfstring.len(),
        tokens
            .iter()
            .map(|token| token.len())
            .fold(0usize, |a, b| a + b)
    );
    let tools = ToolPrompt {
        list: vec![
            Tool {
                name_for_model: "add".to_string(),
                name_for_human: "addition".to_string(),
                description_for_model: "Calculate the result of addition.like 3+3=6".to_string(),
                schema: schema_add,
                call: Box::new(|s: String| {
                    use std::str::FromStr;
                    let p: Add = serde_json::from_str(&s).unwrap();
                    format!(
                        "{}+{}={}",
                        p.lhs,
                        p.rhs,
                        i32::from_str(&p.lhs).unwrap() + p.rhs
                    )
                }),
            },
            Tool {
                name_for_model: "sub".to_string(),
                name_for_human: "subtraction".to_string(),
                description_for_model: "Calculate the result of subtraction.like 4-1=3".to_string(),
                schema: schema_sub,
                call: Box::new(|s: String| {
                    let p: Sub = serde_json::from_str(&s).unwrap();
                    format!("{}-{}={}", p.lhs, p.rhs, p.lhs - p.rhs)
                }),
            },
        ],
    };
    let gbnf_schema = tools.qwen_gbnf();
    let p = tools.qwen_tool_prompt();
    println!("{}", gbnf_schema);
    println!("{}", p);
    let model = Model::new("qwen2-1_5b-instruct-q2_k.gguf");
    let sampler = Sampler::new(&gbnf_schema);
    let chatreturn = model.chat(
        "Help me calculate the result of one plus one.",
        &[["system", &p]],
        &sampler,
    );
    println!("return:{}", chatreturn);
    let result = tools.qwen_parse_call(&chatreturn);
    println!("result:{}", result);
    let sampler = Sampler::new(&gbnf_schema);
    let chatreturn = model.chat(
        "Help me calculate the result of two minus one.",
        &[["system", &p]],
        &sampler,
    );
    println!("return:{}", chatreturn);
    let result = tools.qwen_parse_call(&chatreturn);
    println!("result:{}", result);
}
