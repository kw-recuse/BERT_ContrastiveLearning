from transformers import LongformerTokenizer, LongformerModel

def load_tokenizer_and_model(model_name):
    try:
        tokenizer = LongformerTokenizer.from_pretrained(model_name) 
        model = LongformerModel.from_pretrained(model_name)
        return tokenizer, model
    except:
        raise ValueError(f"{model_name} does not exist.")