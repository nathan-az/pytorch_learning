_config = dict(
    seed=999,
    dataroot="data/celeba",
    # seems to be a bug setting cpu workers >0 on Windows
    workers=0,
    batch_size=128,
    image_size=64,
    nc=3,
    # Size of z latent vector (i.e. size of generator input),
    # nz: size of latent vector z, ngf and ndf are num feature maps in generator and discriminator respectively,
    nz=100,
    ngf=64,
    ndf=64,
    num_epochs=5,
    lr=0.0002,
    beta1=0.5,
    n_gpus=1,
)


def get_config():
    return _config
