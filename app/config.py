import torch
import os

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
MAX_LEN = 512
GENRE_COLUMNS = ['comedy', 'cult', 'flashback', 'historical', 'revenge', 'romantic', 'scifi', 'violence']
# The artifact path should be relative to the project root when running from there
MODEL_PATH = os.path.join("artifacts", "roberta_best.pt")

# Thresholds from Grid Search
THRESHOLDS = {
    'comedy': 0.25,
    'cult': 0.46,
    'flashback': 0.47,
    'historical': 0.78,
    'revenge': 0.33,
    'romantic': 0.40,
    'scifi': 0.79,
    'violence': 0.34
}
