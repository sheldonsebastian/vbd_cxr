# %% --------------------
from torch.utils import tensorboard

# %% --------------------
writer = tensorboard.SummaryWriter(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr/7_POC/tensorboard/epocher")

# %% --------------------
epochs = 100

step_size = epochs // 10
counter = 1

# %% --------------------
for i in range(epochs):
    # save value at every epoch in tensorboard
    writer.add_scalar("loss", i * 0.5, global_step=i)

    # save value at counter in tensorboard
    if i % step_size == 0:
        writer.add_scalar("mAP", i * 2, global_step=counter)
        counter += 1
