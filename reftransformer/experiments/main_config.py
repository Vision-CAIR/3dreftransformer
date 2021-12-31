DEFAULT_CONFIG = {
    "batch_size": 32,
    "init_lr": 0.0005,

    "scannet_file": '../scans.pkl',
    "referit3D_file": '../nr3d.csv',

    "log_dir": None,
    "checkpoint_dir": None,
    "tensorboard_dir": None,

    "resume_path": None,
    "vocab_file": None,

    "max_distractors": 51,
    "max_seq_len": 24,
    "max_test_objects": 88,
    "max_train_epochs": 50,

    "min_word_freq": 3,

    "mode": "train",

    "n_workers": 2,

    "obj_cls_alpha": 0.5,
    "lang_cls_alpha": 0.5,

    "patience": 10,

    "points_per_object": 1024,
    "random_seed": 2020,

    "s_vs_n_weight": None,

    "word_dropout": 0.1,
    'mentions_target_class_only': True,
    'augment_with_sr3d':None
}
