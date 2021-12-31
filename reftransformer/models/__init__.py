from termcolor import colored

from reftransformer.models.backbone.dgcnn import DGCNN
from reftransformer.models.backbone.lstm_encoder import LSTMEncoder
from reftransformer.models.backbone.mlp import MLP
from reftransformer.models.backbone.word_embeddings import load_glove_pretrained_embedding, make_pretrained_embedding

try:
    from reftransformer.models.backbone.point_net_pp import PointNetPP
except ImportError:
    PointNetPP = None
    msg = colored('Pnet++ is not found. Hence you cannot run all models. Install it via '
                  'external_tools (see README.txt there).', 'red')
    print(msg)
