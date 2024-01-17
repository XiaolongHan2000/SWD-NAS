import torch
from model_search import Network as Network

iteration = 50
model = Network(16, 10, 5, None)
# model = model.cuda()

input = torch.randn(1, 3, 32, 32)
# input = input.cuda()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for _ in range(50):
    _ = model(input)

times = torch.zeros(iteration)
with torch.no_grad():
    for iter in range(iteration):
        starter.record()
        _ = model(input)
        ender.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        times[iter] = starter.elapsed_time(ender)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

