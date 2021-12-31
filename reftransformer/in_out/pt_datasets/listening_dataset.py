import json

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import partial

from transformers import BertTokenizer

from .utils import dataset_to_dataloader, max_io_workers

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import check_segmented_object_order, sample_scan_object, pad_samples, objects_bboxes
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform
from ...data_generation.nr3d import decode_stimulus_string

RELATIONS = {
    'above': 0,
    'below': 1,
    'front': 2,
    'right': 3,
    'left': 4,
    'back': 5
    # 'farthest': 4,
    # 'closest': 0,
}


class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, max_seq_len, points_per_object, max_distractors, seed,
                 class_to_idx=None, object_transformation=None,
                 visualization=True):
        print("dataset created")

        self.references = references
        self.scans = scans
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        self.rnd = np.random.RandomState(seed)
        self.seed = seed

        with open('all_scene_graphs.json') as fin:
            self.scene_graph = json.load(fin)

        # Read the pose information
        with open('../obj_orientation.json') as fin:
            self.obj_orientation = json.load(fin)

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def re_initialize(self):
        if self.rnd != np.random:
            self.rnd = np.random.RandomState(self.seed)
            print("reintializing the random state")

    def get_objects_pose(self, scan_id, obj_list):
        ret = []

        for o in obj_list:
            pose = self.obj_orientation.get(scan_id + '_{}'.format(o.object_id), 16)
            if pose != 16:
                pose = int(pose / (22.5))
            ret.append(pose)

        return ret

    def get_objects_bboxes(self, obj_list):
        ret_pos = []
        ret_scale = []

        for o in obj_list:
            bbox = o.get_bbox()

            ret_pos.append([bbox.cx, bbox.cy, bbox.cz])
            ret_scale.append([bbox.lx, bbox.ly, bbox.lz])

        return ret_pos, ret_scale

    def sample_distractor_utterances(self, scene_utterances, target_stimulus_id, target_instance_type):
        # Get all distractor utterances
        distractor_utterances = scene_utterances[scene_utterances.stimulus_id != target_stimulus_id]

        # Get same-class and different class distractors
        same_class_distractor_utterances = distractor_utterances[
            distractor_utterances.instance_type == target_instance_type]
        different_class_distractor_utternaces = distractor_utterances[
            distractor_utterances.instance_type != target_instance_type]

        # You should sample here at random 10 examples where at most there are 5 same-class distractors and 5
        # different-class distractors
        n = 0
        if len(same_class_distractor_utterances) > 0:
            n = min(30, len(same_class_distractor_utterances))

            if len(different_class_distractor_utternaces) == 0:
                n = 50
            sampled_utterances = list(
                same_class_distractor_utterances.sample(n=n, replace=n < len(same_class_distractor_utterances)).tokens)

        if len(different_class_distractor_utternaces) > 0:
            n = 50 - n
            replace = n > len(different_class_distractor_utternaces)
            sampled_utterances.extend(
                list(different_class_distractor_utternaces.sample(n=n, replace=replace).tokens))

        while len(sampled_utterances) < 50:
            sampled_utterances.append([])

        res = []
        res_mask = []
        for el in sampled_utterances:
            tokens_len = len(el) + 2
            res.append(np.array(self.vocab.encode(el, self.max_seq_len, add_begin_end=True),
                                dtype=np.long))

            d = np.ones(self.max_seq_len + 2)
            d[tokens_len:] = 0
            res_mask.append(d)

        return (np.array(res), np.array(res_mask))

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        add_begin_end = True
        tokens = np.array(self.vocab.encode(ref['tokens'], self.max_seq_len, add_begin_end=add_begin_end),
                          dtype=np.long)
        is_nr3d = ref['dataset'] == 'nr3d'
        tokens_len = len(ref['tokens']) + 2 * add_begin_end

        # Get distractor utterances
        scene_id = ref['scan_id']
        # scene_utterances = self.references[self.references.scan_id == scene_id]
        # distractor_utterances = self.sample_distractor_utterances(self.references, ref.stimulus_id, ref.instance_type)

        return scan, target, tokens, is_nr3d, tokens_len, None

    def prepare_distractors(self, scan, target):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]

        # Then all more objects up to max-number of distractors
        already_included = {target_label}
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        self.rnd.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        self.rnd.shuffle(distractors)

        return distractors

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, is_nr3d, tokens_len, distractor_utterances = self.get_reference_data(index)

        # Make a context of distractors
        context = self.prepare_distractors(scan, target)

        # Add target object in 'context' list
        target_pos = self.rnd.randint(len(context) + 1)
        context.insert(target_pos, target)

        # sample point/color for them
        samples = np.array([sample_scan_object(o, self.points_per_object, self.rnd) for o in context])

        # mark their classes
        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)

        if self.object_transformation is not None:
            samples = self.object_transformation(samples)

        res['context_size'] = len(samples)

        # take care of padding, so that a batch has same number of N-objects across scans.
        res['objects'] = pad_samples(samples, self.max_context_size)

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]

        res['target_class'] = self.class_to_idx[target.instance_label]
        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['is_nr3d'] = is_nr3d
        res['scan_id'] = scan.scan_id

        res['objects_mask'] = np.ones(self.max_context_size)
        res['objects_mask'][len(context):] = 0
        res['language_mask'] = np.ones(self.max_seq_len + 2)
        res['language_mask'][tokens_len:] = 0

        if self.visualization:
            distrators_pos = np.zeros((6))  # 6 is the maximum context size we used in dataset collection
            object_ids = np.zeros((self.max_context_size))
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['distrators_pos'] = distrators_pos
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id

        # # tokenize the negative examples
        # negative_text = distractor_utterances[0]
        # negative_text_masks = distractor_utterances[1]

        # res['negative_text_tokens'] = torch.tensor(negative_text)
        # res['negative_text_tokens_mask'] = torch.tensor(negative_text_masks)
        res['utterance'] = self.references.loc[index]['utterance']

        # add the scene graph
        # get the mapping of object id to object position in the tensor (becuase of shuffling)
        mapping = {}
        for ii, el in enumerate(context):
            mapping[el.object_id] = ii

        a = []
        b = []
        r = []
        if scan.scan_id in self.scene_graph:
            scene_graph = self.scene_graph[scan.scan_id]
            for obj_pair, rels in scene_graph.items():
                for rel in rels:
                    if rel not in RELATIONS:
                        continue

                    o_1 = eval(obj_pair)[0]
                    o_2 = eval(obj_pair)[1]
                    if o_1 not in mapping or o_2 not in mapping:
                        continue

                    i_a = mapping[o_1]
                    i_b = mapping[o_2]
                    a.append(i_a)
                    b.append(i_b)
                    r.append(RELATIONS[rel])

        MAX_RELS = 200
        for k in range(MAX_RELS - len(a) + 1):
            a.append(0)
            b.append(0)
            r.append(len(RELATIONS))

        a = np.array(a)
        b = np.array(b)
        r = np.array(r)

        # Reduce to 200 relations if more than that
        if len(a) > MAX_RELS:
            def unison_shuffled_copies(a, b, c):
                assert len(a) == len(b) == len(c)
                p = self.rnd.permutation(len(a))
                return a[p], b[p], c[p]

            res['rel_a'], res['rel_b'], res['rel'] = unison_shuffled_copies(a, b, r)

        res['rel_a'] = a[:MAX_RELS]
        res['rel_b'] = b[:MAX_RELS]
        res['rel'] = r[:MAX_RELS]

        # Get the pose information for the annotated objects
        poses = self.get_objects_pose(scan.scan_id, context)

        bboxes, bboxes_scales = self.get_objects_bboxes(context)

        for i in range(len(context) + 1, self.max_context_size + 1):
            bboxes.append([0, 0, 0])
            bboxes_scales.append([0, 0, 0])
            poses.append(16)
        assert len(bboxes) == self.max_context_size

        res['bboxes'] = np.array(bboxes)
        res['poses'] = np.array(poses, dtype=int)
        res['abs_pos'] = np.concatenate([bboxes, bboxes_scales], axis=1)

        # print(np.array(res['rel_a']).shape, np.array(res['rel_b']).shape, np.array(res['rel']).shape)
        return res


def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=False)
    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == 'test':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            print("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        dataset = ListeningDataset(references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate',
                                   seed=args.random_seed if split == 'test' else None)

        seed = None
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, seed=seed)

    return data_loaders
