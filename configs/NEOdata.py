# Script defining configs for the NEO data

data_path = "/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR/train_val_set"
save_path = "/mnt/c/Users/Imesh/Desktop/summer_proj/models"
height = 512
width = 512

device = 'cuda'
epochs = 15
batch_size = 4
num_classes = 3

optimizer = dict(
        opt = "Adam",
        lr = 1e-3,
        momentum = 0.9,
        weight_decay=1e-4
        )

img_norms = dict(
        means = [],  # red, green, blue, NIR, ....
        stds = []
        )

RGB_classes = dict(
        Nodata = [0, 0, 0],#[155,155,155],
        Water = [58, 221, 254],
        Mangrove = [66,242,30]
        )

