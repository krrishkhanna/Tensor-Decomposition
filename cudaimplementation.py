import torch.backends.cudnn as cudnn
import torch

cudnn.benchmark = True  # Speeds up inference on fixed input sizes
cudnn.enabled = True  # Enables cuDNN optimizations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")