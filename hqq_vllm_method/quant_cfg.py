from hqq.core.quantize import BaseQuantizeConfig


# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q2_config = BaseQuantizeConfig(nbits=2, group_size=64)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64) 
    q8_config = BaseQuantizeConfig(nbits=8, group_size=64)
    
    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q8_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config
        # if i <= n_layers / 2:
        #     quant_config[f'model.layers.{i}.mlp.down_proj'] = q8_config
        # else:
        #     quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config
        
    return quant_config