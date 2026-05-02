import torch
import torch.nn as nn
from model import SahilGPT
from config import *
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

with open("data/train.txt","r",encoding="utf-8") as f:
    text = f.read()

ids = tokenizer.encode(text).ids

data = torch.tensor(ids)

model = SahilGPT().to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

loss_fn = nn.CrossEntropyLoss()

print("Training started...")

for epoch in range(EPOCHS):

    for i in range(0, len(data)-BLOCK_SIZE, BATCH_SIZE):

        x = data[i:i+BLOCK_SIZE].unsqueeze(0).to(DEVICE)
        y = data[i+1:i+BLOCK_SIZE+1].unsqueeze(0).to(DEVICE)

        logits = model(x)

        loss = loss_fn(
            logits.view(-1, VOCAB_SIZE),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss {loss.item()}")

torch.save(model.state_dict(), MODEL_PATH)

print("Training complete.")