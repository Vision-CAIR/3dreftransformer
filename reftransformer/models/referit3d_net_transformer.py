import torch
import argparse
from torch import nn
from collections import defaultdict

from transformers import BertConfig
# from transformers.modeling_bert import BertLayerNorm

from . import DGCNN
from .backbone.memory_meshed_transformer.encoders import MemoryAugmentedEncoder
from .backbone.memory_meshed_transformer.memory_attention import ScaledDotProductAttentionMemory
from .backbone.visual_transformer import CONFIGS, Experiment, VisualTransformerEncoder, WordPositionalEncoding, \
    TextBert, MMT, _get_mask
from .default_blocks import *
from .utils import get_siamese_features
from ..in_out.vocabulary import Vocabulary

try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None


class ReferIt3DPPNet(nn.Module):
    """
    A neural listener for segmented 3D scans based on graph-convolutions.
    """

    def __init__(self,
                 args,
                 object_encoder,
                 language_encoder,
                 transformer,
                 object_language_clf,
                 object_clf=None,
                 language_clf=None):
        """
        Parameters have same meaning as in Base3DListener.

        @param args: the parsed arguments
        @param object_encoder: encoder for each segmented object ([point-cloud, color]) of a scan
        @param language_encoder: encoder for the referential utterance
        @param transformer: the graph net encoder (DGCNN is the used graph encoder)
        given geometry is the referred one (typically this is an MLP).
        @param object_clf: classifies the object class of the segmented (raw) object (e.g., is it a chair? or a bed?)
        @param language_clf: classifies the target-class type referred in an utterance.
        @param object_language_clf: given a fused feature of language and geometry, captures how likely it is that the
        """

        super().__init__()

        self.args = args

        # The language fusion method (either before the graph encoder, after, or in both ways)
        self.language_fusion = args.language_fusion

        # Encoders
        self.object_encoder = object_encoder

        if self.args.experiment == Experiment.COMMON_SPACE_VISUAL_LANG_TOKENS:
            self.language_encoder = language_encoder[0]
            self.language_embedder = language_encoder[1]
            self.language_pos_encoder = language_encoder[2]
            self.lang_embedding_transformation = nn.Linear(args.word_embedding_dim, args.language_latent_dim)
        elif self.args.experiment == Experiment.CONCATENATED_TOKENS:
            self.language_encoder = language_encoder
            self.language_features_projection_layer = nn.Linear(self.args.language_latent_dim,
                                                                self.args.dim)
        elif self.args.experiment in [Experiment.TRANSFORMER_OBJECT_TOKENS_ONLY,
                                      Experiment.TRANSFORMER_OBJECT_LSTM_TOKENS]:
            self.language_encoder = language_encoder
            self.language_features_projection_layer = nn.Linear(self.args.language_latent_dim,
                                                                self.args.dim)
        elif self.args.experiment == Experiment.M4C:
            self.language_encoder = language_encoder
            self.object_features_projection_layer = nn.Linear(self.args.dim,
                                                              self.args.dim)
            self.object_features_layer_norm = BertLayerNorm(self.args.dim)
            self.object_features_dropout = nn.Dropout(self.args.object_features_dropout)

            TEXT_BERT_HIDDEN_SIZE = 768
            self.language_features_projection_layer = nn.Identity()
            if self.args.dim != TEXT_BERT_HIDDEN_SIZE:
                self.language_features_projection_layer = nn.Linear(TEXT_BERT_HIDDEN_SIZE,
                                                                    self.args.dim)
        elif self.args.experiment == Experiment.MEMORY_MESH_TRANSFORMER_COMMON_SPACE:
            self.language_encoder = language_encoder[0]
            self.language_embedder = language_encoder[1]
            self.language_pos_encoder = language_encoder[2]
            self.lang_embedding_transformation = nn.Linear(args.word_embedding_dim, args.language_latent_dim)
        else:
            raise NotImplementedError

        self.transformer = transformer

        # Classifier heads
        self.object_clf = object_clf
        self.language_clf = language_clf
        self.object_language_clf = object_language_clf

    def __call__(self, batch: dict) -> dict:
        # Get feature for utterance
        if self.args.experiment == Experiment.M4C:
            return self.forward_m4c(batch)
        elif self.args.experiment == Experiment.CONCATENATED_TOKENS:
            return self.forward_concatenated_tokens(batch)
        elif self.args.experiment == Experiment.COMMON_SPACE_VISUAL_LANG_TOKENS:
            return self.forward_common_space(batch)
        elif self.args.experiment == Experiment.MEMORY_MESH_TRANSFORMER_COMMON_SPACE:
            return self.forward_memory_mesh(batch)
        elif self.args.experiment == Experiment.TRANSFORMER_OBJECT_TOKENS_ONLY:
            return self.forward_object_tokens_transformer(batch)
        elif self.args.experiment == Experiment.TRANSFORMER_OBJECT_LSTM_TOKENS:
            return self.forward_object_lstm_tokens_transformer(batch)
        else:
            raise NotImplementedError

    def forward_m4c(self, batch):
        result = defaultdict(lambda: None)

        # Get features for each segmented scan object based on color and point-cloud
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim

        # Classify the segmented objects
        if self.object_clf is not None:
            objects_classifier_features = objects_features
            result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        batch['text'] = batch['text'].cuda()
        batch['text_len'] = batch['text_len'].cuda()
        # Get the lang tokens bert embeddings
        lang_token_mask = _get_mask(
            batch['text_len'], batch['text'].size(1)
        )

        lang_token_embeddings = self.language_encoder(
            txt_inds=batch['text'],
            txt_mask=lang_token_mask
        )

        lang_token_embeddings = self.language_features_projection_layer(lang_token_embeddings)

        # Create a mask for the object embeddings
        object_token_mask = _get_mask(batch['num_objects'], objects_features.size(1)).cuda()

        mmt_input_objects_features = self.object_features_projection_layer(objects_features)
        mmt_input_objects_features = self.object_features_dropout(mmt_input_objects_features)

        # Ready to be passed to the MM transformer
        mmt_results = self.transformer(
            txt_emb=lang_token_embeddings,
            txt_mask=lang_token_mask,
            obj_emb=mmt_input_objects_features,
            obj_mask=object_token_mask)

        cls_features = mmt_results['mmt_txt_output'][:, 0,
                       ...]  # B x 768 Take the CLS token TODO check that we are having one CLS token
        cls_features_expanded = torch.unsqueeze(cls_features, -1). \
            expand(-1, -1, object_token_mask.size(1)).transpose(2, 1)  # B X N_Objects x lang-latent-dim
        final_features = torch.cat([objects_features, cls_features_expanded], dim=-1)

        result['logits'] = get_siamese_features(self.object_language_clf, final_features, torch.cat)

        return result

    def forward_concatenated_tokens(self, batch):
        result = defaultdict(lambda: None)

        # Get features for each segmented scan object based on color and point-cloud
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim

        # Classify the segmented objects
        if self.object_clf is not None:
            objects_classifier_features = objects_features
            result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        n_objects = batch['objects'].size(1)
        lang_features = self.language_encoder(batch['tokens'])

        # Classify the target instance label based on the text
        if self.language_clf is not None:
            result['lang_logits'] = self.language_clf(lang_features)

        objects_mask = batch['objects_mask']

        x = objects_features
        y = lang_features
        x_mask = objects_mask
        y_mask = None

        final_features = self.transformer(x, x_mask, y, y_mask)

        if self.args.cls_head == 'dot_softmax':
            lang_features = self.language_features_projection_layer(lang_features)
            result['logits'] = get_siamese_features(self.object_language_clf, (final_features, lang_features, x_mask),
                                                    torch.stack, independent_dim=0)
        else:
            result['logits'] = get_siamese_features(self.object_language_clf, final_features, torch.cat)

        return result

    def forward_object_tokens_transformer(self, batch):
        result = defaultdict(lambda: None)

        # Get features for each segmented scan object based on color and point-cloud
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim

        # Classify the segmented objects
        if self.object_clf is not None:
            objects_classifier_features = objects_features
            result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        n_objects = batch['objects'].size(1)
        lang_features = self.language_encoder(batch['tokens'])

        # Classify the target instance label based on the text
        if self.language_clf is not None:
            result['lang_logits'] = self.language_clf(lang_features)

        objects_mask = batch['objects_mask']

        x = objects_features
        y = None
        x_mask = objects_mask
        y_mask = None

        final_features = self.transformer(x, x_mask, y, y_mask)

        if self.args.cls_head == 'dot_softmax':
            lang_features = self.language_features_projection_layer(lang_features)
            # Should I prject the object tokens TODO ask ?
            result['logits'] = get_siamese_features(self.object_language_clf, (final_features, lang_features, x_mask),
                                                    torch.stack, independent_dim=0)
        else:
            result['logits'] = get_siamese_features(self.object_language_clf, final_features, torch.cat)

        return result

    def forward_common_space(self, batch):
        result = defaultdict(lambda: None)

        # Get features for each segmented scan object based on color and point-cloud
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim

        # Classify the segmented objects
        if self.object_clf is not None:
            objects_classifier_features = objects_features
            result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        n_objects = batch['objects'].size(1)
        lang_features = self.language_encoder(batch['tokens'])

        # Classify the target instance label based on the text
        if self.language_clf is not None:
            result['lang_logits'] = self.language_clf(lang_features)

        objects_mask = batch['objects_mask']
        language_mask = batch['language_mask']

        assert self.language_pos_encoder is not None and self.language_embedder is not None
        language_embeddings = self.language_pos_encoder(
            self.language_embedder(batch['tokens']).transpose(0, 1)).transpose(0, 1)
        language_embeddings = self.lang_embedding_transformation(language_embeddings)

        x = objects_features
        x_mask = objects_mask
        y = language_embeddings
        y_mask = language_mask

        final_features = self.transformer(x, x_mask, y, y_mask)

        if self.args.cls_head == 'dot_softmax':
            # Get the language token mean feature to represent the whole sentence
            lang_tokens = final_features[:, n_objects:, :].mean(dim=1)

            # Then pass this to the model
            result['logits'] = get_siamese_features(self.object_language_clf,
                                                    (final_features[:, :n_objects, :], lang_tokens, x_mask),
                                                    torch.stack, independent_dim=0)
        else:
            result['logits'] = get_siamese_features(self.object_language_clf, final_features[:, :n_objects, :],
                                                    torch.cat)

        return result

    def forward_memory_mesh(self, batch):
        result = defaultdict(lambda: None)

        # Get features for each segmented scan object based on color and point-cloud
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim

        # Classify the segmented objects
        if self.object_clf is not None:
            objects_classifier_features = objects_features
            result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        n_objects = batch['objects'].size(1)
        lang_features = self.language_encoder(batch['tokens'])

        # Classify the target instance label based on the text
        if self.language_clf is not None:
            result['lang_logits'] = self.language_clf(lang_features)

        objects_mask = batch['objects_mask']
        language_mask = batch['language_mask']

        assert self.language_pos_encoder is not None and self.language_embedder is not None
        language_embeddings = self.language_pos_encoder(
            self.language_embedder(batch['tokens']).transpose(0, 1)).transpose(0, 1)
        language_embeddings = self.lang_embedding_transformation(language_embeddings)

        x = objects_features
        x_mask = objects_mask
        y = language_embeddings
        y_mask = language_mask

        mask = 1 - torch.cat([x_mask, y_mask],
                             dim=1)  # x_mask, y_mask here are of shape B X N_objects, B X N_language_tokens, Here True indicates masking.
        mask  = mask.unsqueeze(1).unsqueeze(1).bool()
        input = torch.cat([x, y], dim=1)  # B x [N_objects + N_language_tokens] x feat_dim
        final_features = self.transformer((input, mask))

        if self.args.cls_head == 'dot_softmax':
            # Get the fatures of the last transformer layer
            final_features, _mask = final_features
            final_features = final_features[:, -1, :, :]
            # Get the language token mean feature to represent the whole sentence
            lang_tokens = final_features[:, n_objects:, :].mean(dim=1)

            # Then pass this to the model
            result['logits'] = get_siamese_features(self.object_language_clf,
                                                    (final_features[:, :n_objects, :], lang_tokens, x_mask),
                                                    torch.stack, independent_dim=0)
        else:
            result['logits'] = get_siamese_features(self.object_language_clf, final_features[:, :n_objects, :],
                                                    torch.cat)

        return result

    def forward_object_lstm_tokens_transformer(self, batch):

        result = defaultdict(lambda: None)

        # Get features for each segmented scan object based on color and point-cloud
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim

        # Classify the segmented objects
        if self.object_clf is not None:
            objects_classifier_features = objects_features
            result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        n_objects = batch['objects'].size(1)
        lang_features = self.language_encoder(batch['tokens'])  # B x language_latet_dim

        # Classify the target instance label based on the text
        if self.language_clf is not None:
            result['lang_logits'] = self.language_clf(lang_features)

        objects_mask = batch['objects_mask']

        x = objects_features
        y = lang_features.unsqueeze(1)
        x_mask = objects_mask
        y_mask = torch.ones((lang_features.size(0), 1)).cuda()

        final_features = self.transformer(x, x_mask, y, y_mask)

        if self.args.cls_head == 'dot_softmax':
            lang_features = self.language_features_projection_layer(lang_features)
            # Should I prject the object tokens TODO ask ?
            result['logits'] = get_siamese_features(self.object_language_clf,
                                                    (final_features[:, :n_objects, :], lang_features, x_mask),
                                                    torch.stack, independent_dim=0)
        else:
            result['logits'] = get_siamese_features(self.object_language_clf, final_features, torch.cat)

        return result


def instantiate_referit3d_pp(args: argparse.Namespace, vocab: Vocabulary, n_obj_classes: int) -> nn.Module:
    """
    Creates a neural listener by utilizing the parameters described in the args
    but also some "default" choices we chose to fix in this paper.

    @param args:
    @param vocab:
    @param n_obj_classes: (int)
    """

    # convenience
    if args.experiment == Experiment.M4C:
        geo_out_dim = args.dim
    else:
        geo_out_dim = args.object_latent_dim
        lang_out_dim = args.language_latent_dim

    # make an object (segment) encoder for point-clouds with color
    if args.object_encoder == 'pnet_pp':
        object_encoder = single_object_encoder(geo_out_dim)
    else:
        raise ValueError('Unknown object point cloud encoder!')

    # Optional, make a bbox encoder
    object_clf = None
    if args.obj_cls_alpha > 0:
        print('Adding an object-classification loss.')
        object_clf = object_decoder_for_clf(geo_out_dim, n_obj_classes)

    language_clf = None
    if args.lang_cls_alpha > 0 and args.experiment != Experiment.M4C:
        print('Adding a text-classification loss.')
        language_clf = text_decoder_for_clf(lang_out_dim, n_obj_classes)
        # typically there are less active classes for text, but it does not affect the attained text-clf accuracy.

    # make a language encoder
    if args.experiment == Experiment.M4C:
        bert_config = BertConfig(num_hidden_layers=3)
        lang_encoder = TextBert.from_pretrained("bert-base-uncased", config=bert_config)
    else:
        lang_encoder = token_encoder(vocab=vocab,
                                     word_embedding_dim=args.word_embedding_dim,
                                     lstm_n_hidden=lang_out_dim,
                                     word_dropout=args.word_dropout,
                                     random_seed=args.random_seed)

    if args.experiment == Experiment.COMMON_SPACE_VISUAL_LANG_TOKENS:
        language_embedder = nn.Embedding(len(vocab), args.word_embedding_dim, padding_idx=vocab.pad)
        language_pos_encoder = WordPositionalEncoding(args.word_embedding_dim, max_len=args.max_seq_len + 2)
        lang_encoder = (lang_encoder, language_embedder, language_pos_encoder)

    if args.experiment == Experiment.MEMORY_MESH_TRANSFORMER_COMMON_SPACE:
        language_embedder = nn.Embedding(len(vocab), args.word_embedding_dim, padding_idx=vocab.pad)
        language_pos_encoder = WordPositionalEncoding(args.word_embedding_dim, max_len=args.max_seq_len + 2)
        lang_encoder = (lang_encoder, language_embedder, language_pos_encoder)

    #
    # Transformer model
    #
    if args.experiment == Experiment.M4C:
        bert_config = BertConfig(hidden_size=768, num_hidden_layers=4)
        transformer = MMT(config=bert_config)
    elif args.experiment == Experiment.MEMORY_MESH_TRANSFORMER_COMMON_SPACE:
        transformer = MemoryAugmentedEncoder(args.num_layers, padding_idx=0,
                                             attention_module=ScaledDotProductAttentionMemory,
                                             d_in=args.dim,
                                             d_model=args.dim,
                                             d_ff=args.ffn_dim,
                                             h=args.num_heads,
                                             d_k=args.dim,
                                             d_v=args.dim, attention_module_kwargs={'m': args.m})
    else:
        transformer = VisualTransformerEncoder()

    if args.cls_head == 'dot_softmax':
        object_language_clf = object_lang_clf_softmax(in_dim=args.dim)
    else:
        object_language_clf = object_lang_clf(in_dim=args.dim)

    model = ReferIt3DPPNet(
        args=args,
        object_encoder=object_encoder,
        language_encoder=lang_encoder,
        transformer=transformer,
        object_clf=object_clf,
        language_clf=language_clf,
        object_language_clf=object_language_clf, )

    return model
