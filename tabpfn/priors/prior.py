from torch.utils.data import DataLoader


class PriorDataLoader(DataLoader):
    pass
    # init accepts num_steps as first argument

    # has two attributes set on class or object level:
    # num_features: int and
    # num_outputs: int
    # fuse_x_y: bool
    # Optional: validate function that accepts a transformer model
