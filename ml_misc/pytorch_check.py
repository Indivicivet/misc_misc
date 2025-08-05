import torch


class PlusMinusModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        return (self.param ** 2 * x + (-self.param)).sum()  # use param twice


model = PlusMinusModule()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

opt.zero_grad()
loss = model(torch.ones(5))
loss.backward()
opt.step()
print(f"{loss.item()=}, {model.param=}, {model.param.grad=}")

opt.zero_grad()
loss = model(torch.ones(5))  # necessary, otherwise get error
loss.backward()
opt.step()
print(f"{loss.item()=}, {model.param=}, {model.param.grad=}")
