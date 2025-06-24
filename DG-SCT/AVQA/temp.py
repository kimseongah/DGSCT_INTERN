from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
model = AutoModel.from_pretrained("google-bert/bert-large-uncased")

