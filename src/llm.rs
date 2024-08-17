use cxx::let_cxx_string;
use regex::Regex;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;


pub use schemars::{JsonSchema,schema_for,schema::RootSchema};
pub use serde_json::*;
pub use serde::Deserialize;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("localllm/llama.cpp/common/grammar-parser.h");
        include!("localllm/llama.cpp/include/llama.h");
        include!("localllm/include/myparser.h");
        type llama_model;
        type llama_context;
        type llama_sampling_context;
        unsafe fn load_model(modelpath:&CxxString,init_backend:bool,model:*mut *mut llama_model,context:*mut *mut llama_context,debug:bool);
        unsafe fn free_model(model:*mut llama_model,context:*mut llama_context,free_backend:bool);
        unsafe fn generate(model:*mut llama_model,context:*mut llama_context,prompt:&CxxString,history:&[[&str;2]],sampling:*mut llama_sampling_context,debug:bool)->UniquePtr<CxxString>;
        unsafe fn init_sampler(ctx_sampling:*mut *mut llama_sampling_context,grammar:&CxxString,debug:bool);
        unsafe fn free_sampler(ctx_sampling:*mut llama_sampling_context);
        fn json_schema_to_gbnf(src:&CxxString)->UniquePtr<CxxString>;
    }
}


pub fn init(modelpath:&Path,prompt:&str,history:&[[&str;2]],grammar:&str)->String {
    let model = Model::new(modelpath);
    let sampler = Sampler::new(grammar);
    model.chat(prompt, history, &sampler)
}

static MODEL_COUNT:AtomicU32 = AtomicU32::new(0);

pub struct Model(*mut ffi::llama_model,*mut ffi::llama_context);

impl Model {
    pub fn new<P:AsRef<Path>>(modelpath:P)->Model{
        let_cxx_string!(modelpath = modelpath.as_ref().to_string_lossy().to_string());
        let mut model: *mut ffi::llama_model = std::ptr::null_mut();
        let mut context: *mut ffi::llama_context = std::ptr::null_mut();
        unsafe {
            ffi::load_model(&modelpath, MODEL_COUNT.fetch_add(1, Ordering::SeqCst)==1, &mut model, &mut context,false);
        };
        Model(model,context)
    }
    pub fn chat(&self,prompt:&str,history:&[[&str;2]],sampler:&Sampler)->String{
        let_cxx_string!(prompt = prompt);
        unsafe {
            ffi::generate(self.0, self.1, &prompt, history, sampler.0,false)
        }.to_string()
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            ffi::free_model(self.0, self.1, MODEL_COUNT.fetch_sub(1, Ordering::SeqCst)==0);
        }
    }
}



pub struct Sampler(*mut ffi::llama_sampling_context);

impl Sampler {
    pub fn new(grammar:&str)->Sampler {
        let mut ctx_sampling: *mut ffi::llama_sampling_context = std::ptr::null_mut();
        let_cxx_string!(grammar = grammar);
        unsafe {
            ffi::init_sampler(&mut ctx_sampling, &grammar,false);
        }
        Sampler(ctx_sampling)
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            ffi::free_sampler(self.0);
        }
    }
}

pub struct Tool{
    pub name_for_model:String,
    pub name_for_human:String,
    pub description_for_model:String,
    pub schema:RootSchema,
    pub call:Box<dyn Fn(String)->String>,
}

pub struct ToolPrompt{
    pub list:Vec<Tool>,
}

impl ToolPrompt {
    pub fn qwen_tool_prompt(&self)->String{
        let tools = self.list.iter().map(|Tool{name_for_model,name_for_human,description_for_model,schema,call:_}|{
            let mut schema = schema.schema.clone();
            schema.metadata = None;
            let schema = serde_json::to_string(&schema).unwrap();
            format!(
r#"### {name_for_human}

{name_for_model}: {description_for_model} Parameters：{schema}
"#                
            )
            }).collect::<Vec<String>>().join("");
        let toollist = self.list.iter().map(|tool|tool.name_for_model.as_str()).collect::<Vec<&str>>().join(",");
format!(r#"
# Tools

## You have access to the following tools:

{tools}

## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

✿FUNCTION✿: The tool to use, should be one of [{toollist}]
✿ARGS✿: The input of the tool
✿RESULT✿: The result returned by the tool. The image needs to be rendered as ![](url)
✿RETURN✿: Reply based on tool result"#

)

    }

    pub fn qwen_gbnf(&self)->String{
        let mut rules = vec![];
        let mut rootoneof = vec![];
        for tool in &self.list {
            let gsub = grammar_with_namespace(&tool.schema,&tool.name_for_model);
            rules.push(gsub);
            rootoneof.push(format!(r#""{}\n✿ARGS✿:" {}-root"#,tool.name_for_model,tool.name_for_model));
        }
        let rootrule = format!(r#"root ::= "✿FUNCTION✿:" ( {} )"#,rootoneof.join(" | "));
        rules.push(rootrule);
        rules.join("\n")
    }
    pub fn qwen_parse_call(&self,return_value:&str)->String {
        let (one,two) = return_value.trim().split_once("\n").unwrap();
        let function_name = one.strip_prefix("✿FUNCTION✿:").unwrap().trim();
        let function_params = two.strip_prefix("✿ARGS✿:").unwrap().trim();
        self.list.iter().find_map(|tool|if tool.name_for_model==function_name {
            Some((&tool.call)(function_params.to_string()))
        }
        else {
            None
        }).unwrap()
    }
}

pub fn json_schema_to_gbnf(src:&str)->String {
    let_cxx_string!(src=src);
    ffi::json_schema_to_gbnf(&src).to_string()
}

pub fn tokens(src:&str)->Vec<String> {
    static TOKEN : OnceLock<Regex> = OnceLock::new();
    let regex = TOKEN.get_or_init(||{
        Regex::new(r#"#[^\n]+|[a-z-0-9]+|::=|[\s]+|\[(\\[\s\S]|[^\]])*\]|"(\\[\s\S]|[^"])*"|\{[0-9]+(,[0-9]+|)\}|[\|\(\)\[\]?+\.\*]"#).unwrap()
    });
    regex.find_iter(src).map(|m|m.as_str().to_string()).collect()
}


fn grammar_with_namespace(schema:&RootSchema,namespace:&str)->String {
    let g = json_schema_to_gbnf(&serde_json::to_string(schema).unwrap());
    let mut tokens = tokens(&g);
    assert_eq!(g.len(),tokens.iter().map(|token|token.len()).fold(0usize, |a,b|a+b));
    static IDENT : OnceLock<Regex> = OnceLock::new();
    let regex = IDENT.get_or_init(||{
        Regex::new(r#"^[a-z-0-9]+$"#).unwrap()
    });
    if !regex.is_match(namespace) {
        panic!("namespace: {namespace} not ^[a-z-0-9]+$");
    }
    for token in &mut tokens {
        if regex.is_match(&token) {
            let newtoken = format!("{}-{}",namespace,token);
            token.clear();
            token.push_str(&newtoken);
        }
    }
    tokens.join("")
}