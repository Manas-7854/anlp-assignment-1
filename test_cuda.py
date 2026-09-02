import torch

# Check if CUDA is available for PyTorch
print("CUDA Available:", torch.cuda.is_available())

# Check the specific CUDA version PyTorch is built with
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)

