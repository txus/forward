import inspect
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
from safetensors.torch import save_file

PIPE = False

if PIPE:
    pipeline = pipeline(task="text-generation", model="meta-llama/Llama-3.2-1B", device_map='cpu', dtype=torch.bfloat16)
    model = pipeline.model
    model.set_attn_implementation("eager")
else:
    tokenizer = AutoTokenizer.from_pretrained("../model")
    model = AutoModelForCausalLM.from_pretrained("../model", dtype=torch.bfloat16, device_map='cpu', attn_implementation="eager")


for name, module in model.named_modules():
    print(name)

# model.layers
# model.layers.0
# model.layers.0.self_attn
# model.layers.0.self_attn.q_proj
# model.layers.0.self_attn.k_proj
# model.layers.0.self_attn.v_proj
# model.layers.0.self_attn.o_proj
# model.layers.0.mlp
# model.layers.0.mlp.gate_proj
# model.layers.0.mlp.up_proj
# model.layers.0.mlp.down_proj
# model.layers.0.mlp.act_fn
# model.layers.0.input_layernorm
# model.layers.0.post_attention_layernorm

activations = {}

def get_activation(name):
    def hook(module, input, output):
        #print(f"{name} ({output[0].dtype}): {output[0]}")
        activations[name] = output[0].detach().unsqueeze(0)
    return hook
 
model.model.embed_tokens.register_forward_hook(get_activation('embed_tokens'))

# Register hooks at different layers
# For Llama models, the structure is: model.model.layers[i]
for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(get_activation(f'layers.{i}'))

    layer.self_attn.q_proj.register_forward_hook(get_activation(f'layers.{i}.self_attn.q_proj'))
    layer.self_attn.k_proj.register_forward_hook(get_activation(f'layers.{i}.self_attn.k_proj'))
    layer.self_attn.v_proj.register_forward_hook(get_activation(f'layers.{i}.self_attn.v_proj'))
    layer.self_attn.o_proj.register_forward_hook(get_activation(f'layers.{i}.self_attn.o_proj'))

    #layer.mlp.register_forward_hook(get_activation(f'layers.{i}.mlp'))

    layer.mlp.gate_proj.register_forward_hook(get_activation(f'layers.{i}.mlp.gate_proj'))
    layer.mlp.up_proj.register_forward_hook(get_activation(f'layers.{i}.mlp.up_proj'))
    layer.mlp.down_proj.register_forward_hook(get_activation(f'layers.{i}.mlp.down_proj'))
    layer.mlp.act_fn.register_forward_hook(get_activation(f'layers.{i}.mlp.act_fn'))
    layer.input_layernorm.register_forward_hook(get_activation(f'layers.{i}.input_layernorm'))
    layer.post_attention_layernorm.register_forward_hook(get_activation(f'layers.{i}.post_attention_layernorm'))

# Run inference
if PIPE:
    print('pipeline: ', pipeline("The capital of france is", max_new_tokens=1, temperature=0))
else:
    inputs = tokenizer(["The capital of france is"], return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        print("ALL OGITS", outputs.logits)
        logits = outputs.logits[:, -1, :]
        print("LAST TOKEN LOGITS", logits)

        activations['logits'] = logits.detach().unsqueeze(0)

        tok_id = torch.argmax(logits, dim=-1)
    print(tok_id)
    print(tokenizer.batch_decode(tok_id))

#print('generate: ', model.generate(**inputs))

# Now activations dict contains all the intermediate values
activations_cpu = {k: v.cpu().to(torch.bfloat16).contiguous() for k, v in activations.items()}
save_file(activations_cpu, "activations.safetensors")
print(f"Saved {len(activations_cpu)} tensors")
