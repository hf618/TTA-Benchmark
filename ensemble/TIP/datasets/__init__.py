from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars


dataset_list = {
                "OxfordPets": OxfordPets,
                "eurosat": EuroSAT,
                "UCFf101": UCF101,
                "SUN397": SUN397,
                "Caltech101": Caltech101,
                "DTD": DescribableTextures,
                "fgvc_aircraft": FGVCAircraft,
                "Food101": Food101,
                "Flower102": OxfordFlowers,
                "StanfordCars": StanfordCars,
                }


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)