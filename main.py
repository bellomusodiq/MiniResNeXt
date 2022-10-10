import torch
import torch.nn as nn

from .models import ResNeXt

CUDA = torch.cuda.is_available()

model = ResNext()

if CUDA:
    model = model.cuda()

# hyper parameters
learning_rate = 0.01
decay_rate = 0.5
batch_size = 100
num_epochs = 60
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()


model_path = "./mini_resnext.pt"

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

if __name__ == __main__:
    for epoch in range(num_epochs):
        train_loss, train_accuracy = epoch_train(
            model, num_epochs, epoch, train_loader, loss_fn, optimizer
        )
        test_loss, test_accuracy = epoch_test(
            model, num_epochs, epoch, test_loader, loss_fn
        )
        torch.save(model.state_dict(), model_path)
