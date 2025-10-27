# Script defining configs for the NEO data

data_path = "/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR/train_val_set"
save_path = "/mnt/c/Users/Imesh/Desktop/summer_proj/models"

device = 'cuda'
epochs = 15
batch_size = 4
num_workers = 6
num_classes = 3

optimizer = dict(
        opt = "Adam",
        lr = 1e-3,
        momentum = 0.9,
        weight_decay=1e-4
        )

img_norms = dict(
        means = [417.9286, 405.0440, 415.4319],  # red, green, blue, NIR, ....
        stds = [102.1648,  75.8841,  60.6446]
        )

RGB_classes = dict(
        Nodata = [0, 0, 0],#[155,155,155],
        Water = [58, 221, 254],
        Mangrove = [66,242,30]
        )

#TODO create dict for max and min values of each band 



