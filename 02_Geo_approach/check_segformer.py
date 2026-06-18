import numpy as np, matplotlib.pyplot as plt

d = np.load(r"...patches_train_A\patches\pos_0000000.npz")
print(list(d.keys()), d["dsm"].shape)
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(d["dsm"])
ax[0].set_title("dsm")
ax[1].imshow(d["water"])
ax[1].set_title("water")
ax[2].imshow(d["label"])
ax[2].set_title("label")
ax[3].imshow(d["tpi_r5"])
ax[3].set_title("tpi_r5")
plt.show()
