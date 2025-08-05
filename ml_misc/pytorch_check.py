import torch


class PlusMinusModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        return (self.param ** 2 * x + (-self.param)).sum()                                  # scalar loss


model = PlusMinusModule()
loss = model(torch.ones(5))
loss.backward(retain_graph=True)  # no retain graph => error
loss.backward()
# loss.backward()  # if re-called => error
print(f"{loss.item()=}, {model.param=}, {model.param.grad=}")
