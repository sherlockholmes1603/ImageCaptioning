import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from models import CaptioningModel  # Placeholder for model import
from Dataloader import flickr8k_dataloader  # Placeholder for Dataloader import

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 128
NUM_WORKERS = 8
EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
LEARNING_RATE = 3e-5
MAX_EPOCHS = 100

# Load data
root_folder = "path_to_images"  # Update with actual path
caption_file = "path_to_captions.csv"  # Update with actual path
transform = None  # Define appropriate transformations

train_loader, dataset = flickr8k_dataloader(
    root_folder, caption_file, transform, train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)
vocab_size = len(dataset.vocab)

# Initialize model, loss, and optimizer
model = CaptioningModel(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=vocab_size, num_layers=NUM_LAYERS).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# TensorBoard setup
writer = SummaryWriter()

# Training loop
for epoch in range(MAX_EPOCHS):
    model.train()
    for idx, (imgs, captions, lengths) in enumerate(train_loader):
        imgs = imgs.to(device)
        captions = captions.to(device)
        
        outputs = model(imgs, captions, lengths)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if idx % 10 == 0:
            print(f"Epoch [{epoch}/{MAX_EPOCHS}], Step [{idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
            writer.add_scalar("Training Loss", loss.item(), epoch * len(train_loader) + idx)

writer.close()
torch.save(model.state_dict(), "model_checkpoint.pth")
