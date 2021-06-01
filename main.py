import json, os
from attrdict import AttrDict
from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_QUESTION_ANSWERING,
    init_logger,
    set_seed
)

config_dir = 'config'
task = 'news'
config_file = 'deploy_model.json'

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
model = MODEL_FOR_QUESTION_ANSWERING[args.model_type].from_pretrained(
    args.model_name_or_path,
    config=config,
)

# args.device = torch.device('cuda')
# model.to(args.device)


# import the IrisClassifier class defined above
from single_mrc_model import SingleMRCModel

# Create a iris classifier service instance
mrc_service = SingleMRCModel()

# Pack the newly trained model artifact
mrc_service.pack('model', model)

# Save the prediction service to disk for model serving
saved_path = mrc_service.save()