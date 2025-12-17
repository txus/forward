import inspect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from safetensors.torch import save_file

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", dtype=torch.bfloat16, device_map='cpu')

for name, module in model.named_modules():
    print(name)

layer = model.model.layers[0]
print(inspect.getsource(layer.self_attn.__class__))

exit(0)

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
        print(f"{name}: {output[0]}")
        activations[name] = output[0].detach().unsqueeze(0)
    return hook
 
model.model.embed_tokens.register_forward_hook(get_activation('embed_tokens'))

# Register hooks at different layers
# For Llama models, the structure is: model.model.layers[i]
for i, layer in enumerate(model.model.layers):
    #layer.register_forward_hook(get_activation(f'layers.{i}'))
    
    # You can also go deeper into each layer:
    #layer.self_attn.register_forward_hook(get_activation(f'layers.{i}.self_attn'))

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

    break

# Run inference
inputs = tokenizer(["The capital of france is"], return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Now activations dict contains all the intermediate values
activations_cpu = {k: v.cpu().to(torch.bfloat16).contiguous() for k, v in activations.items()}
save_file(activations_cpu, "activations.safetensors")
print(f"Saved {len(activations_cpu)} tensors")
