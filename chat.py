import torch
import torch.nn.functional as F
from model import SahilGPT
from config import *
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

model = SahilGPT().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("SahilGPT ready. Type exit to quit.")

while True:

    user = input("You: ")

    if user == "exit":
        break

    prompt = f"User: {user}\nBot:"

    ids = tokenizer.encode(prompt).ids

    x = torch.tensor(ids).unsqueeze(0).to(DEVICE)

    for _ in range(150):

        logits = model(x[:, -BLOCK_SIZE:])

        probs = F.softmax(logits[:, -1, :] / 0.7, dim=-1)

        next_id = torch.multinomial(probs,1)

        x = torch.cat([x,next_id],dim=1)

    print("Bot:", tokenizer.decode(x[0].tolist()).split("Bot:")[-1])