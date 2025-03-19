from transformers import AutoTokenizer, AutoModel

def load_tokenizer_and_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name) 
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model
    except:
        raise ValueError(f"{model_name} does not exist.")