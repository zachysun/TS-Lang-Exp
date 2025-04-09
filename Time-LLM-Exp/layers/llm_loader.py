from transformers import (
    AutoModel, AutoTokenizer,
    LlamaConfig,
    GPT2Config,
    BertConfig,
    DistilBertConfig,
    RobertaConfig,
    OPTConfig,
    # PhiConfig,
    # GemmaConfig,
    GPT2Model, GPT2Tokenizer
)


def LLMLoader(configs):
    config_map = {
        'LLAMA': ('huggyllama/llama-7b', LlamaConfig),  # dim: 4096, layer: 32
        'GPT2': ('openai-community/gpt2', GPT2Config),  # dim: 768, layer: 12
        'BERT': ('google-bert/bert-base-uncased', BertConfig),  # dim: 768, layer: 12
        'DistilBERT': ('distilbert/distilbert-base-uncased', DistilBertConfig),  # dim: 768, layer: 6
        'RoBERTa': ('FacebookAI/xlm-roberta-base', RobertaConfig),  # dim: 768, layer: 12
        'OPT-125m': ('facebook/opt-125m', OPTConfig),  # dim: 768, layer: 12
        'OPT-350m': ('facebook/opt-350m', OPTConfig),
        'OPT-1.3B': ('facebook/opt-1.3b', OPTConfig),
        # 'Phi-1': ('microsoft/phi-1', PhiConfig),  # dim: 2048, layer: 24
        # 'Gemma-2B': ('google/gemma-2b', GemmaConfig),  # dim: 3072, layer: 28
        'GPT2_Scratch': ('GPT2_Scratch', GPT2Config)
    }

    model_name, config_cls = config_map[configs.llm_model]
    if model_name == 'GPT2_Scratch':
        gpt2_config = GPT2Config(
            num_hidden_layers=configs.llm_layers,
            output_attentions=True,
            output_hidden_states=True
        )
        model = GPT2Model(config=gpt2_config)
        tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
    else:
        if config_cls:
            config = config_cls.from_pretrained(model_name)
            config.num_hidden_layers = configs.llm_layers
            config.output_attentions = True
            config.output_hidden_states = True
        else:
            config = None
        try:
            model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, local_files_only=True)
        except EnvironmentError:
            print(f"Local model files for {configs.llm_model} not found. Attempting to download...")
            model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, local_files_only=False)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        except EnvironmentError:
            print(f"Local tokenizer files for {configs.llm_model} not found. Attempting to download...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)

        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': pad_token})
            tokenizer.pad_token = pad_token

    return model, tokenizer
