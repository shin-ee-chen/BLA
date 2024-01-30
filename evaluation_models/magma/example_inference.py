from magma import Magma
from magma.image_input import ImageInput

model = Magma.from_checkpoint(
    config_path = "/home/xchen/magma/configs/MAGMA_v1.yml",
    checkpoint_path = "/projects/0/gusr0608/models/magma_model/mp_rank_00_model_states.pt",
    device = 'cuda:0'
)

inputs =[
    ## supports urls and path/to/image
    ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
    'Q:Is the sentence the man holds the woman appropriate for this image? yes or no?A:'
]

## returns a tensor of shape: (1, 149, 4096)
embeddings = model.preprocess_inputs(inputs)  

## returns a list of length embeddings.shape[0] (batch size)
output = model.generate(
    embeddings = embeddings,
    max_steps = 6,
    temperature = 0.7,
    top_k = 0,
)  

print(output[0]) ##  A cabin on a lake
