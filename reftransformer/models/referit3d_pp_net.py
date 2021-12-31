import torch
import argparse
from torch import nn
import numpy as np
from collections import defaultdict

from transformers import BertConfig, BertTokenizer

from .backbone.attention import MultiHeadAttention, MultiHeadAttentionRelPosEmb
from .backbone.visual_transformer import WordPositionalEncoding
from .default_blocks import *
from .utils import get_siamese_features
from ..in_out.vocabulary import Vocabulary

try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None


class ReferIt3DPPNetT(nn.Module):
    """
    A neural listener for segmented 3D scans based on transformers.
    """

    def __init__(self,
                 args,
                 object_encoder,
                 language_encoder,
                 transformer,
                 object_language_clf,
                 vocab,
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
        # Encoders
        self.object_encoder = object_encoder
        self.obj_t = ObjTransformer(args)

        self.lang_t = language_encoder[0]
        self.language_embedder = language_encoder[1]
        self.language_pos_encoder = language_encoder[2]

        # Classifier heads
        self.object_clf = object_clf
        self.transformer = transformer
        self.language_clf = language_clf

        # Dot Scale normalization
        self.object_language_clf = object_language_clf
        if args.neg_loss:
            print("USING NEG LOSS")
            self.positive_obj_negative_lang_head = ObjectLangSoftmaxClassifier(dim=args.hidden_dim)
        else:
            print("NOT USING NEG LOSS")
        if args.rel_loss:
            print("USING REL LOSS")
            self.relation_classifier = nn.Sequential(
                nn.Linear(2 * args.hidden_dim, 32),
                nn.Linear(32, 7, bias=True)
            )
        else:
            print("NOT USING REL LOSS")

        self.args = args

        if self.args.use_object_context_fc:
            self.object_context_fc = nn.Sequential(
                nn.Linear(args.hidden_dim * 2, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU()
            )
        else:
            self.object_context_fc = None

        #
        # Pose prediction 16 pins
        #
        if self.args.pose_loss:
            self.pose_cls = nn.Sequential(
                nn.Linear(args.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 16),  # Number of pose bins
            )
        else:
            self.pose_cls = None

        self.vocab = vocab

        if self.args.lang_masked_loss:
            self.masked_lang_clf = nn.Sequential(
                nn.Conv1d(in_channels=args.hidden_dim, out_channels=64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=64, out_channels=len(self.vocab), kernel_size=1)
            )

    def random_lang_feat(self, tokens):
        """
        :param feats: n_words
        :return:
        """
        masked_feats = tokens
        feat_mask = torch.zeros((len(tokens), 1), dtype=torch.float32)

        for i in range(len(tokens)):
            prob = np.random.random()

            # mask token with probability
            if prob < self.args.lang_mask_rate:
                # Need to predict this feat
                feat_mask[i] = 1.

                prob /= self.args.lang_mask_rate  # make it in scale [0, 1.0]

                # 80% randomly change token to MASK token
                if prob < 0.8:
                    masked_feats[i] = self.vocab('<mask>')
                    continue

                # 10% randomly change token to random token
                if prob < 0.9:
                    assert self.vocab is not None
                    random_token = np.random.choice(list(self.vocab.word2idx.keys()), 1)[0]
                    masked_feats[i] = self.vocab(random_token)
                    continue
                else:
                    # -> rest 10% randomly keep current feat
                    pass

        return masked_feats, feat_mask

    def __call__(self, batch: dict) -> dict:
        result = {}
        bs = batch['objects'].size(0)

        # Get features for each segmented scan object based on color and point-cloud
        pnet_objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                     aggregator=torch.stack)  # B X N_Objects x object-latent-dim
        trans_objects_features = self.obj_t(x=pnet_objects_features, x_mask=batch['objects_mask'],
                                            x_pos=batch['bboxes'],
                                            batch=batch)
        if self.object_context_fc is not None:
            assert self.args.use_object_context_fc
            objects_features = get_siamese_features(self.object_context_fc,
                                                    torch.cat([pnet_objects_features, trans_objects_features], dim=-1),
                                                    aggregator=torch.stack)
        else:
            objects_features = trans_objects_features

        # Classify the segmented objects
        # Get the (a, b) object pair feauters
        if self.args.rel_loss:
            rel_logits = []
            for i in range(objects_features.size(0)):
                obj_features = objects_features[i]
                a = obj_features.index_select(0, batch['rel_a'][i])
                b = obj_features.index_select(0, batch['rel_b'][i])
                c = torch.cat([a, b], dim=1)
                rel_logits.append(self.relation_classifier(c))

            result['rel_logits'] = torch.stack(rel_logits)
        result['class_logits'] = get_siamese_features(self.object_clf, objects_features, torch.stack)

        if self.pose_cls is not None:
            result['poses_logits'] = get_siamese_features(self.pose_cls, objects_features, torch.stack)

        # Language token, supposed to be the CLS token only for the provided utterances
        if self.args.lang_masked_loss and self.training:
            x_new = []
            x_masked_token = []

            for i in range(batch['tokens'].size(0)):
                t = self.random_lang_feat(batch['tokens'][i, ...])

                x_new.append(t[0])
                x_masked_token.append(t[1])

            lang_inp, lang_inp_mask = torch.stack(x_new), torch.stack(x_masked_token).cuda()
        else:
            lang_inp = batch['tokens']
            lang_inp_mask = torch.zeros_like(batch['language_mask']).cuda()

        language_embeddings = self.language_pos_encoder(
            self.language_embedder(lang_inp).transpose(0, 1)).transpose(0, 1)
        language_features = self.lang_t(language_embeddings, batch['language_mask'])  # B x 26 x 128

        # Classify the target instance label based on the text
        if self.args.use_cls_token_only:
            result['lang_logits'] = self.language_clf(language_features[:, 0, ...])
        else:
            result['lang_logits'] = self.language_clf(language_features.mean(dim=1))

        # Get the language features for each negative example
        # Transformer
        x = objects_features
        y = language_features  # B x 1 x 768
        x_mask = batch['objects_mask']
        y_mask = batch['language_mask']

        n_objects = x.size(1)
        final_features = self.transformer(x, x_mask, y, y_mask)

        final_object_features = final_features[:, :n_objects, ...]

        if self.args.use_cls_token_only:
            final_language_features = final_features[:, n_objects, ...]  # positive CLS token enhanced
        else:
            final_language_features = final_features[:, n_objects:, ...].mean(
                dim=1)  # positive CLS token enhanced

        if self.args.lang_masked_loss and self.training:
            result['masked_lang_logits'] = self.masked_lang_clf(
                final_features[:, n_objects:, ...].transpose(2, 1)).transpose(2, 1)
            result['lang_inp_mask'] = lang_inp_mask

        result['logits'] = get_siamese_features(
            self.object_language_clf,
            (final_object_features, final_language_features, x_mask),
            torch.stack, independent_dim=0)

        if self.training and self.args.neg_loss:
            if self.args.same_softmax:

                if self.args.neg_lang_level > 0:
                    #
                    # Target object feature to negative language
                    #
                    neg_lang_logits = torch.zeros(bs, bs).cuda().float()  # B x 16

                    for b_i in range(bs):
                        # Get the neg lang features
                        neg_lang_features = [final_language_features[b_i]]
                        neg_lang_features.extend(final_language_features[
                                                     batch['neg_lang_mask'][
                                                         b_i].cuda().bool()])
                        neg_lang_features = torch.stack(neg_lang_features)

                        n = len(neg_lang_features)

                        # Get the ground truth target object feature
                        target_object_feature = final_object_features[b_i, batch['target_pos'][b_i], ...]  # 1 x 16

                        neg_lang_logits[b_i, :n] = self.object_language_clf(
                            (target_object_feature.unsqueeze(0), neg_lang_features, None))
                        neg_lang_logits[b_i, n:] = -1000000  # masking

                    result.update({
                        'neg_lang_logits': neg_lang_logits,
                    })

                if self.args.neg_obj_level > 1:
                    #
                    # Target utterance to negative objects in the scene
                    #
                    max_obj_seq = self.args.max_distractors + 1

                    neg_obj_logits = torch.zeros(bs, max_obj_seq * bs).cuda().float()  # B x (B * max_obj_seq)

                    for b_i in range(bs):
                        # Get the neg object features
                        neg_obj_features = [final_object_features[b_i, batch['target_pos'][b_i], ...]]
                        neg_obj_features.extend(final_object_features[batch['neg_obj_mask'][b_i].cuda().bool()])
                        neg_obj_features = torch.stack(neg_obj_features)

                        n = len(neg_obj_features)

                        target_object_utterance = final_language_features[b_i, ...]

                        neg_obj_logits[b_i, :n] = self.object_language_clf(
                            (target_object_utterance.unsqueeze(0), neg_obj_features, None))
                        neg_obj_logits[b_i, n:] = -1000000  # masking

                    result.update({
                        'neg_obj_logits': neg_obj_logits
                    })

        return result


class ReferIt3DPPNetO(nn.Module):
    """
    A neural listener for segmented 3D scans based on transformers.
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
        # Encoders
        self.object_encoder = object_encoder
        self.language_encoder = language_encoder
        self.transformer = transformer

        # Classifier heads
        self.object_clf = object_clf
        self.language_clf = language_clf

        # Dot Scale normalization
        self.object_language_clf = object_language_clf
        if args.neg_loss:
            print("USING NEG LOSS")
            self.positive_obj_negative_lang_head = ObjectLangSoftmaxClassifier(dim=args.hidden_dim)
        else:
            print("NOT USING NEG LOSS")
        if args.rel_loss:
            print("USING REL LOSS")
            self.relation_classifier = nn.Sequential(
                nn.Linear(2 * args.hidden_dim, 32),
                nn.Linear(32, 7, bias=True)
            )
        else:
            print("NOT USING REL LOSS")

        self.args = args

    def __call__(self, batch: dict) -> dict:
        result = defaultdict(lambda: None)

        # Get features for each segmented scan object based on color and point-cloud
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim

        # Classify the segmented objects
        objects_classifier_features = objects_features
        result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        # Language token, supposed to be the CLS token only for the provided utterances
        language_features = self.language_encoder(batch['tokens'])

        # Classify the target instance label based on the text
        result['lang_logits'] = self.language_clf(language_features)

        # Get the language features for each negative example
        if self.training and self.args.neg_loss:
            batch_size = batch['negative_text_tokens'].size(0)

            negative_examples_cls_features = []
            for i in range(batch_size):
                text_tokens = batch['negative_text_tokens'][i].cuda()  # 50 x 26
                # text_mask = batch['negative_text_tokens_mask'][i].cuda()  # 50 x 26
                negative_feature = self.language_encoder(text_tokens)  # 50  x 768
                negative_examples_cls_features.append(negative_feature)
            negative_examples_cls_features = torch.stack(negative_examples_cls_features, dim=0)  # B x 50 x768

            # Transformer
            x = objects_features
            y = torch.cat([language_features.unsqueeze(1), negative_examples_cls_features], dim=1)  # B x 11 x 768
            x_mask = batch['objects_mask']
            y_mask = torch.ones(x.size(0), 51).cuda()  # B x 11

            n_objects = x.size(1)
            final_features = self.transformer(x, x_mask, y, y_mask)

            final_object_features = final_features[:, :n_objects, ...]
            final_positive_language_features = final_features[:, n_objects, ...]  # positive CLS token enhanced

            # Calculate the logits
            all_languages = final_features[:, n_objects:, ...]  # B x 11 x 768
            positive_obj_negative_lang = []
            for i in range(batch_size):
                # Get the positive (target) object
                target_pos = batch['target_pos'][i].item()
                positive_object_feature = final_object_features[i, target_pos, ...]  # 768

                # Calculate the similarity
                positive_obj_negative_lang.append(
                    self.positive_obj_negative_lang_head(
                        (all_languages[i, :], positive_object_feature, torch.ones((51)).cuda()))
                )

            result['positive_obj_negative_lang'] = torch.stack(positive_obj_negative_lang)  # B x 11 x 768
            result['gt_positive_obj_negative_lang'] = torch.zeros(
                batch_size).cuda().long()  # B always the best language is the first

            result['logits'] = get_siamese_features(
                self.object_language_clf,
                (final_object_features, final_positive_language_features, x_mask),
                torch.stack, independent_dim=0)  # B x 11

            # Get the (a, b) object pair feauters
            if self.args.rel_loss:
                rel_logits = []
                for i in range(final_object_features.size(0)):
                    obj_features = final_object_features[i]
                    a = obj_features.index_select(0, batch['rel_a'][i])
                    b = obj_features.index_select(0, batch['rel_b'][i])
                    c = torch.cat([a, b], dim=1)
                    rel_logits.append(self.relation_classifier(c))

                result['rel_logits'] = torch.stack(rel_logits)

            return result

        else:
            # Transformer
            x = objects_features
            y = language_features.unsqueeze(1)  # B x 1 x 768
            x_mask = batch['objects_mask']
            y_mask = torch.ones(x.size(0), 1).cuda()

            n_objects = x.size(1)
            final_features = self.transformer(x, x_mask, y, y_mask)

            final_object_features = final_features[:, :n_objects, ...]
            final_positive_language_features = final_features[:, n_objects, ...]  # positive CLS token enhanced

            result['logits'] = get_siamese_features(
                self.object_language_clf,
                (final_object_features, final_positive_language_features, x_mask),
                torch.stack, independent_dim=0)

            # Get the (a, b) object pair feauters
            if self.args.rel_loss:
                rel_logits = []
                for i in range(final_object_features.size(0)):
                    obj_features = final_object_features[i]
                    a = obj_features.index_select(0, batch['rel_a'][i])
                    b = obj_features.index_select(0, batch['rel_b'][i])
                    c = torch.cat([a, b], dim=1)
                    rel_logits.append(self.relation_classifier(c))

                result['rel_logits'] = torch.stack(rel_logits)

            return result


class ReferIt3DPPNetBL(nn.Module):
    """
    A neural listener for segmented 3D scans based on transformers.
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
        # Encoders
        self.object_encoder = object_encoder
        self.language_encoder = language_encoder
        self.transformer = transformer

        # Classifier heads
        self.object_clf = object_clf
        self.language_clf = language_clf

        # Dot Scale normalization
        self.object_language_clf = object_language_clf
        self.args = args

    def __call__(self, batch: dict) -> dict:
        result = defaultdict(lambda: None)

        # Get features for each segmented scan object based on color and point-cloud
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim

        # Classify the segmented objects
        objects_classifier_features = objects_features
        result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        # Language token, supposed to be the CLS token only for the provided utterances
        language_features = self.language_encoder(batch['tokens'])

        # Classify the target instance label based on the text
        result['lang_logits'] = self.language_clf(language_features)

        language_features_expanded = torch.unsqueeze(language_features, -1).expand(-1, -1,
                                                                                   objects_features.size(1)).transpose(
            2, 1)  # B X N_Objects x lang-latent-dim
        final_features = torch.cat([objects_features, language_features_expanded], dim=2)

        result['logits'] = get_siamese_features(self.object_language_clf, final_features, torch.cat)

        return result


class TEncoderBlock(nn.Module):
    """
    https://github.com/majumderb/rezero/blob/master/rezero/transformer/rztx.py
    """

    def __init__(self, args):
        super().__init__()

        embed_size = args.hidden_dim
        heads = args.num_heads
        forward_expansion = args.forward_expansion
        attn_dropout: float = args.dropout_in_attn
        dropout: float = args.dropout_in_ff

        print("Embed size:{}".format(embed_size))
        print("Number of heads in self attention:{}".format(heads))
        print("Attention dropout:{}".format(attn_dropout))
        print("FeedForward dropout:{}".format(dropout))

        self.args = args

        #
        # Self attention stage
        #
        self.attention = MultiHeadAttention(dim=embed_size, num_heads=heads, dropout_rate=attn_dropout)
        self.dropout_1 = nn.Dropout(dropout)

        #
        # Normalization method
        #
        self.layer_norm_1 = self.layer_norm_2 = self.rezero_weight_1 = self.rezero_weight_2 = None

        if not self.args.use_rezero:
            self.layer_norm_1 = nn.LayerNorm(embed_size)
            self.layer_norm_2 = nn.LayerNorm(embed_size)
        else:
            self.rezero_weight_1 = nn.Parameter(torch.Tensor([0]))
            self.rezero_weight_2 = nn.Parameter(torch.Tensor([0]))

        #
        # Feed forward stage
        #
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion, embed_size)
        )
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, rel_pos=None):
        # Multi head attention
        attention = self.attention(q=query, k=key, v=value, mask=mask)

        # Add & Norm
        if not self.args.use_rezero:
            x = self.layer_norm_1(self.dropout_1(attention) + query)
        else:
            x = self.dropout_1(self.rezero_weight_1 * attention) + query

        # Feed Forward
        forward = self.feed_forward(x)

        # Add & Norm
        if not self.args.use_rezero:
            out = self.layer_norm_2(self.dropout_2(forward) + x)
        else:
            out = self.dropout_2(self.rezero_weight_2 * forward) + x

        return out


class ObjTEncoderBlock(nn.Module):
    """
    https://github.com/majumderb/rezero/blob/master/rezero/transformer/rztx.py
    """

    def __init__(self, args):
        super().__init__()

        embed_size = args.hidden_dim
        heads = args.num_heads
        forward_expansion = args.forward_expansion
        attn_dropout: float = args.dropout_in_attn
        dropout: float = args.dropout_in_ff

        print("Embed size:{}".format(embed_size))
        print("Number of heads in self attention:{}".format(heads))
        print("Attention dropout:{}".format(attn_dropout))
        print("FeedForward dropout:{}".format(dropout))

        self.args = args

        #
        # Self attention stage
        #
        self.attention = MultiHeadAttentionRelPosEmb(dim=embed_size, num_heads=heads, dropout_rate=attn_dropout)
        self.dropout_1 = nn.Dropout(dropout)

        #
        # Normalization method
        #
        self.layer_norm_1 = self.layer_norm_2 = self.rezero_weight_1 = self.rezero_weight_2 = None

        if not self.args.use_rezero:
            self.layer_norm_1 = nn.LayerNorm(embed_size)
            self.layer_norm_2 = nn.LayerNorm(embed_size)
        else:
            self.rezero_weight_1 = nn.Parameter(torch.Tensor([0]))
            self.rezero_weight_2 = nn.Parameter(torch.Tensor([0]))

        #
        # Feed forward stage
        #
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion, embed_size)
        )
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, rel_pos):
        # Multi head attention
        attention = self.attention(q=query, k=key, v=value, mask=mask, rel_pos=rel_pos)

        # Add & Norm
        if not self.args.use_rezero:
            x = self.layer_norm_1(self.dropout_1(attention) + query)
        else:
            x = self.dropout_1(self.rezero_weight_1 * attention) + query

        # Feed Forward
        forward = self.feed_forward(x)

        # Add & Norm
        if not self.args.use_rezero:
            out = self.layer_norm_2(self.dropout_2(forward) + x)
        else:
            out = self.dropout_2(self.rezero_weight_2 * forward) + x

        return out


class MultimodalTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        #
        # Transformer encoder blocks
        #
        self.args = args
        self.encoders = nn.ModuleList()
        for i in range(self.args.num_layers):
            self.encoders.append(TEncoderBlock(args))

    def forward(self, x, x_mask, y=None, y_mask=None):
        """
        @param x: features of first modality  (query tokens)
        @param x_mask: [B, num_queries (queries tokens)]
        @param y: features of language (other modality)
        @param y_mask: [B, num_queries (queries here are the language feature tokens)]
        """
        # Create the self attention mask
        # zeros where padding tokens exist
        # ones otherwise

        if y is not None and y_mask is not None:
            mask = torch.cat([x_mask, y_mask],
                             dim=1)
            query = torch.cat([x, y], dim=1)
            key = query
            value = key

            batch_size = query.size(0)
            n_tokens = query.size(1)

            # Make the mask of current size [B, n_keys] to size
            # [batch_size, num_queries (tokens), num_keys]
            mask = mask.unsqueeze(1).repeat(1, n_tokens, 1).view(batch_size, n_tokens, n_tokens)

        elif y is None and y_mask is None:
            mask = x_mask

            query = x
            key = query
            value = key

            batch_size = query.size(0)
            n_tokens = query.size(1)

            # Make the mask of current size [B, n_keys] to size
            # [batch_size, num_queries (tokens), num_keys]
            mask = mask.unsqueeze(1).repeat(1, n_tokens, 1).view(batch_size, n_tokens, n_tokens)
        else:
            raise

        for encoder_block in self.encoders:
            query = encoder_block(query=query, key=key, value=value, mask=mask)

        return query


class ObjTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        #
        # Transformer encoder blocks
        #
        self.args = args
        self.encoders = nn.ModuleList()
        for i in range(2):
            if self.args.rel_pos_emb:
                print("USING REL POS EMB")
                self.encoders.append(ObjTEncoderBlock(args))
            else:
                print("NOT USING REL POS EMB")
                self.encoders.append(TEncoderBlock(args))

        #
        # Relative positional embedding network
        #
        if self.args.rel_pos_emb:
            self.rel_pos_fc = nn.Sequential(
                nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1),
            )

        if self.args.add_absolute_pos:
            self.absolute_pos_fc = nn.Sequential(
                nn.Conv1d(in_channels=6, out_channels=32, kernel_size=1),  # Centers (x, y, z), lx, ly, lz
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            )

    def random_feat(self, feats):
        """
        :param feats: n_objects x 128
        :return:
        """
        mask_feats = feats
        feat_mask = torch.zeros((len(feats), 1), dtype=torch.float32)

        for i in range(len(feats)):
            prob = np.random.random()

            # mask token with probability
            if prob < self.args.obj_mask_rate:
                prob /= self.args.obj_mask_rate  # make it in scale [0, 1.0]

                # 80% randomly change token to zero feat
                if prob < 0.8:
                    # Need to predict this feat
                    feat_mask[i] = 1.

                    mask_feats[i, :] = 0.0
                    continue
                else:
                    # -> rest 20% randomly keep current feat
                    pass

        return mask_feats, feat_mask

    def forward(self, x, x_mask, x_pos, batch, y=None, y_mask=None):
        """
        @param x: features of first modality  (query tokens)
        @param x_mask: [B, num_queries (queries tokens)]
        @param y: features of language (other modality)
        @param y_mask: [B, num_queries (queries here are the language feature tokens)]
        """
        # Create the self attention mask
        # zeros where padding tokens exist
        # ones otherwise
        batch_size = x.size(0)
        n_objects = x.size(1)

        if self.args.rel_pos_emb:
            #
            # Add positional Embedding
            #
            rel_pos = x_pos.unsqueeze(2) - x_pos.unsqueeze(1)  # [B, seq_len, seq_len, position_dim]

            # This will be shared with all the attention heads in all layers
            rel_pos_feat = self.rel_pos_fc(rel_pos.view(batch_size, 3, n_objects * n_objects).float())
            rel_pos_feat = rel_pos_feat.view(batch_size, n_objects, n_objects)
        else:
            rel_pos_feat = None

        #
        # Object masked training
        #
        if self.args.object_masked_loss and self.training:
            x_new = []
            x_masked_token = []

            for i in range(x.size(0)):
                t = self.random_feat(x[i, ...])

                x_new.append(t[0])
                x_masked_token.append(t[1])

            x, x_masked_tokens = torch.stack(x_new), torch.stack(x_masked_token)

        if self.args.add_absolute_pos:
            abs_pos = self.absolute_pos_fc(batch['abs_pos'].transpose(2, 1).float())  # B x 52 x 32
            x = x + abs_pos.transpose(2, 1)

        if y is not None and y_mask is not None:
            mask = torch.cat([x_mask, y_mask],
                             dim=1)
            query = torch.cat([x, y], dim=1)
            key = query
            value = key

            batch_size = query.size(0)
            n_tokens = query.size(1)

            # Make the mask of current size [B, n_keys] to size
            # [batch_size, num_queries (tokens), num_keys]
            mask = mask.unsqueeze(1).repeat(1, n_tokens, 1).view(batch_size, n_tokens, n_tokens)

        elif y is None and y_mask is None:
            mask = x_mask

            query = x
            key = query
            value = key

            batch_size = query.size(0)
            n_tokens = query.size(1)

            # Make the mask of current size [B, n_keys] to size
            # [batch_size, num_queries (tokens), num_keys]
            mask = mask.unsqueeze(1).repeat(1, n_tokens, 1).view(batch_size, n_tokens, n_tokens)
        else:
            raise

        for encoder_block in self.encoders:
            query = encoder_block(query=query, key=key, value=value, mask=mask, rel_pos=rel_pos_feat)

        return query


class LangTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        #
        # Transformer encoder blocks
        #
        self.args = args
        self.encoders = nn.ModuleList()
        for i in range(2):
            self.encoders.append(TEncoderBlock(args))

    def forward(self, x, x_mask, y=None, y_mask=None):
        """
        @param x: features of first modality  (query tokens)
        @param x_mask: [B, num_queries (queries tokens)]
        @param y: features of language (other modality)
        @param y_mask: [B, num_queries (queries here are the language feature tokens)]
        """
        # Create the self attention mask
        # zeros where padding tokens exist
        # ones otherwise

        if y is not None and y_mask is not None:
            mask = torch.cat([x_mask, y_mask],
                             dim=1)
            query = torch.cat([x, y], dim=1)
            key = query
            value = key

            batch_size = query.size(0)
            n_tokens = query.size(1)

            # Make the mask of current size [B, n_keys] to size
            # [batch_size, num_queries (tokens), num_keys]
            mask = mask.unsqueeze(1).repeat(1, n_tokens, 1).view(batch_size, n_tokens, n_tokens)

        elif y is None and y_mask is None:
            mask = x_mask

            query = x
            key = query
            value = key

            batch_size = query.size(0)
            n_tokens = query.size(1)

            # Make the mask of current size [B, n_keys] to size
            # [batch_size, num_queries (tokens), num_keys]
            mask = mask.unsqueeze(1).repeat(1, n_tokens, 1).view(batch_size, n_tokens, n_tokens)
        else:
            raise

        for encoder_block in self.encoders:
            query = encoder_block(query=query, key=key, value=value, mask=mask)

        return query


def instantiate_referit3d_pp_o(args: argparse.Namespace, vocab: Vocabulary, n_obj_classes: int) -> nn.Module:
    """
    Creates a neural listener by utilizing the parameters described in the args
    but also some "default" choices we chose to fix in this paper.

    @param args:
    @param vocab:
    @param n_obj_classes: (int)
    """

    # make an object (segment) encoder for point-clouds with color
    object_encoder = single_object_encoder(args.hidden_dim)
    object_clf = object_decoder_for_clf(args.hidden_dim, n_obj_classes)

    # make a language encoder
    language_encoder = token_encoder(vocab=vocab, word_embedding_dim=args.hidden_dim, lstm_n_hidden=args.hidden_dim,
                                     word_dropout=0.1)
    language_clf = text_decoder_for_clf(args.hidden_dim, n_obj_classes)

    # transformer  model
    transformer = MultimodalTransformer(args=args)

    # The dot product compatibility score
    object_language_clf = ObjectLangSoftmaxClassifier(dim=args.hidden_dim)

    model = ReferIt3DPPNetO(
        args=args,
        object_encoder=object_encoder,
        language_encoder=language_encoder,
        transformer=transformer,
        object_clf=object_clf,
        language_clf=language_clf,
        object_language_clf=object_language_clf, )

    return model


def instantiate_referit3d_pp_t(args: argparse.Namespace, vocab: Vocabulary, n_obj_classes: int) -> nn.Module:
    """
    Creates a neural listener by utilizing the parameters described in the args
    but also some "default" choices we chose to fix in this paper.

    @param args:
    @param vocab:
    @param n_obj_classes: (int)
    """

    # make an object (segment) encoder for point-clouds with color
    object_encoder = single_object_encoder(args.hidden_dim)
    object_clf = object_decoder_for_clf(args.hidden_dim, n_obj_classes)

    language_encoder = LangTransformer(args=args)
    language_embedder = nn.Embedding(len(vocab), 128, padding_idx=vocab.pad)
    language_pos_encoder = WordPositionalEncoding(128, max_len=args.max_seq_len + 2)
    language_clf = text_decoder_for_clf(args.hidden_dim, n_obj_classes)

    # mmt transformer  model
    transformer = MultimodalTransformer(args=args)

    # The dot product compatibility score
    object_language_clf = ObjectLangSoftmaxClassifier(dim=args.hidden_dim)

    model = ReferIt3DPPNetT(
        args=args,
        object_encoder=object_encoder,
        language_encoder=(language_encoder, language_embedder, language_pos_encoder),
        transformer=transformer,
        object_clf=object_clf,
        language_clf=language_clf,
        object_language_clf=object_language_clf,
        vocab=vocab)

    return model


def instantiate_referit3d_pp_bl(args: argparse.Namespace, vocab: Vocabulary, n_obj_classes: int) -> nn.Module:
    """
    Creates a neural listener by utilizing the parameters described in the args
    but also some "default" choices we chose to fix in this paper.

    @param args:
    @param vocab:
    @param n_obj_classes: (int)
    """

    # make an object (segment) encoder for point-clouds with color
    object_encoder = single_object_encoder(args.hidden_dim)
    object_clf = object_decoder_for_clf(args.hidden_dim, n_obj_classes)

    # make a language encoder
    language_encoder = token_encoder(vocab=vocab, word_embedding_dim=args.hidden_dim, lstm_n_hidden=args.hidden_dim,
                                     word_dropout=0.1)
    language_clf = text_decoder_for_clf(args.hidden_dim, n_obj_classes)
    object_language_clf = object_lang_clf(in_dim=args.hidden_dim * 2)

    model = ReferIt3DPPNetBL(
        args=args,
        object_encoder=object_encoder,
        language_encoder=language_encoder,
        transformer=None,
        object_clf=object_clf,
        language_clf=language_clf,
        object_language_clf=object_language_clf, )

    return model


def instantiate_referit3d_pp(args: argparse.Namespace, vocab: Vocabulary, n_obj_classes: int):
    if args.type == 'visual_transformer':
        return instantiate_referit3d_pp_o(args, vocab, n_obj_classes)
    if args.type == 'three_transformers':
        return instantiate_referit3d_pp_t(args, vocab, n_obj_classes)
    if args.type == 'baseline':
        return instantiate_referit3d_pp_bl(args, vocab, n_obj_classes)
