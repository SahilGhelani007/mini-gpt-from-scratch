from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=30000,
    special_tokens=["[PAD]","[UNK]","[EOS]"]
)

tokenizer.train(["data/train.txt"], trainer)

tokenizer.save("tokenizer.json")

print("Tokenizer trained.")