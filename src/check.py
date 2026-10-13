import numpy as np
import pandas as pd
import sklearn
import matplotlib
import torch

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scikit-learn", sklearn.__version__)
print("matplotlib", matplotlib.__version__)
print("torch", torch.__version__)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
print(torch.rand(3, 3))