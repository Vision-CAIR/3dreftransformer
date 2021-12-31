from multiprocessing import Process

import ujson
import time
import os.path as osp
from collections import defaultdict

from reftransformer.data_generation.sr3d.allocentric.allocentric_generator import AllocentricGenerator
from reftransformer.data_generation.sr3d.horizontal_proximity.horizontal_generator import HorizontalProximityGenerator
from reftransformer.data_generation.sr3d.vertical_proximity.vertical_generator import VerticalProximityGenerator
from reftransformer.utils import unpickle_data


class SceneGraph(Process):

    def __init__(self, scenes, start, end):
        super().__init__()
        self.scenes = scenes[start:end]
        self.n = end - start

        # Read the Nr3d target classes
        with open('../data/language/nr3d_target_class_to_idx.json.txt') as fin:
            self.target_classes = ujson.load(fin).keys()

        assert len(self.target_classes) == 76

        self.edge_types = types = ['left', 'right', 'front', 'back', 'above', 'below',
                                   'farthest', 'closest']

    def __call__(self, idx):
        """
        Generate a scene graph where the nodes should contain all the 78 target classes
         found in ReferIt3D Nr3D dataset
        """
        graph = defaultdict(list)
        scene = self.scenes[idx]

        # check if already generated
        if osp.exists(('scene_graphs/{}.json'.format(scene.scan_id))):
            print("ALREADY FOUND: scene:{}".format(scene.scan_id))
            return

        t = time.time()
        print("STARTED: processing scene:{}".format(scene.scan_id))

        scene_objects = scene.three_d_objects
        scene_target_objects = [obj for obj in scene_objects if obj.instance_label in self.target_classes]
        scene_other_objects = [obj for obj in scene_objects if obj.instance_label not in self.target_classes]

        # Try to find the relation between each target object in Nr3D and the other ones
        for target in scene_target_objects:
            for other_object in scene_other_objects:
                key = (target.object_id, other_object.object_id)
                key_inv = (other_object.object_id, target.object_id)

                if key not in graph:
                    t_o_rel, o_t_rel = self.get_relations(scene, target, other_object)
                    if t_o_rel:
                        graph[key] = t_o_rel
                    if o_t_rel:
                        graph[key_inv] = o_t_rel

        for target in scene_target_objects:
            for other_object in scene_other_objects:
                key = (target.object_id, other_object.object_id)

                # Check closest and farthest
                r = HorizontalProximityGenerator.has_horizontal_relation(scene, target, other_object)
                if r is not None:
                    graph[key].append(r)

        # Save the scene graph
        with open('scene_graphs/{}.json'.format(scene.scan_id), 'w') as fout:
            ujson.dump(graph, fout)
        print("Ended: processing scene:{}, elapsed time:{}".format(scene.scan_id, time.time() - t))

    def run(self) -> None:
        for ii in range(self.n):
            self(ii)

    @staticmethod
    def get_relations(scan, target, anchor):
        t_to_a_relations = []
        a_to_t_relations = []

        complement = {
            'left': 'right',
            'right': 'left',
            'front': 'back',
            'back': 'front',
            'above': 'below',
            'below': 'above'
        }

        # Check Left, Right, Front, Back
        allocentric = AllocentricGenerator(verbose=False)
        a = allocentric.has_allocentric_relations(target, anchor)
        if a is not None:
            t_to_a_relations.append(a)
            a_to_t_relations.append(complement[a])

        # Check above or below
        vertical = VerticalProximityGenerator(verbose=False)
        v = vertical.has_vertical_relation(target, anchor)
        if v is not None:
            t_to_a_relations.append(v)
            a_to_t_relations.append(complement[v])

        return sorted(t_to_a_relations), sorted(a_to_t_relations)


if __name__ == '__main__':
    # load all the scans
    scannet, scenes = (unpickle_data('../scans.pkl'))

    scenes_per_process = 26
    n_processes = 28
    process_list = []
    start = 0
    end = scenes_per_process
    for i in range(n_processes):
        process_list.append(SceneGraph(scenes, start, end))
        start = min(len(scenes), start + scenes_per_process)
        end = min(len(scenes), start + scenes_per_process)

    s = time.time()
    for i in range(n_processes):
        process_list[i].start()

    for i in range(n_processes):
        process_list[i].join()
