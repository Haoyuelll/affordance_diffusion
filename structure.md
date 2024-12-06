# Code structure

## Content Net

### GLIDE
- Configs specified in `configs/model/content_glide.yaml`

#### Models
- Transformer
- Diffusion

In training phase, Transformer is freezed by default while diffusion is not.


#### Dataset
- Configs specified in `configs/data/hoi4d_train.yaml`

- GLIDE
  List of datasets. Each contains:    
  ```python
  th.LongTensor(tokens), th.BoolTensor(mask), base_tensor, base_obj, base_mask, mask_param.astype(np.float32).reshape(-1), text

  # or in training
  tokens, masks, reals, inpaint_image, inpaint_mask, mask_param, _ = batch
  ```
  - `tokens` and `text`: tokens for null text, `text=''`. 
  
    See `utils/glide_utils.py` for details. Func `get_tokens_and_mask` for actual prompted version defined but never used.
  
  - `mask_param`: shape (6,0). 
    
    Defined mask types: 'gt', 'lollipop', and 'pose', details in line 246 in `dataset/ho3pairs.py`
  
  - `base_obj` or `inpaint_image`: impainted object-only image 


#### Resume from pretrained model
Both original GLIDE and released model from affordance diffusiom are eligible to load (line 51 in `models/content_glide.py`)


- `base_impaint.pt`: pretrained GLIDE
- `output/release/content_glide/checkpoints/last.ckpt`: released Content Net


