import json, os
from attrdict import AttrDict
from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_QUESTION_ANSWERING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION,
    init_logger,
    set_seed
)

config_dir = 'config'
task = 'news'
config_file = 'IntensiveReadingModule_deploy_model.json'

with open(os.path.join(config_dir, task, config_file)) as f:
    args = AttrDict(json.load(f))

args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

# 모델 불러오기
config = CONFIG_CLASSES[args.model_type].from_pretrained(
    args.model_name_or_path,
)
tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
    args.model_name_or_path,
    do_lower_case=args.do_lower_case,
)
IntensiveReadingModule = MODEL_FOR_QUESTION_ANSWERING[args.model_type].from_pretrained(
    args.model_name_or_path,
    config=config,
)

config_dir = 'config'
task = 'news'
config_file = 'SketchReadingModule_deploy_model.json'

with open(os.path.join(config_dir, task, config_file)) as f:
    args = AttrDict(json.load(f))

args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

# 모델 불러오기
config = CONFIG_CLASSES[args.model_type].from_pretrained(
    args.model_name_or_path,
)
tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
    args.model_name_or_path,
    do_lower_case=args.do_lower_case,
)
SketchReadingModule = MODEL_FOR_SEQUENCE_CLASSIFICATION[args.model_type].from_pretrained(
    args.model_name_or_path,
    config=config,
)

from dual_mrc_model import DualMRCModel

mrc_service = DualMRCModel()

mrc_service.pack('intensive', IntensiveReadingModule)
mrc_service.pack('sketch', SketchReadingModule)

saved_path = mrc_service.save()
