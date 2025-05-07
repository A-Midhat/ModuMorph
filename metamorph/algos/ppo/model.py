import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from collections import OrderedDict
from gym import spaces

from metamorph.config import cfg
from metamorph.utils import model as tu
from metamorph.algos.ppo.transformer import TransformerEncoder, TransformerEncoderLayerResidual


class MLPModel(nn.Module):
    def __init__(self, obs_space, out_dim):
        super(MLPModel, self).__init__()
        
        self.model_args=cfg.MODEL.MLP
        self.seq_len=cfg.MODEL.MAX_LIMBS 
        if not isinstance(obs_space,(spaces.Dict,OrderedDict)) or "proprioceptive" not in obs_space.spaces: 
            raise ValueError("[MLPModel] requires Dict obs_space with 'proprioceptive'.")
            flat_proprio_dim=obs_space["proprioceptive"].shape[0]
            self.output_dim=out_dim
            self.input_layer=nn.Linear(flat_proprio_dim,self.model_args.HIDDEN_DIM)
            self.output_layer=nn.Linear(self.model_args.HIDDEN_DIM,out_dim)
            hidden_input_dim=self.model_args.HIDDEN_DIM
            self.hfield_encoder=None
            hidden_dims=[hidden_input_dim]+[self.model_args.HIDDEN_DIM for _ in range(self.model_args.LAYER_NUM-1)]
            self.hidden_layers=tu.make_mlp_default(hidden_dims)
    
    def forward(self, obs_proprio, obs_env=None, **kwargs):
        
        embedding=self.input_layer(obs_proprio)
        embedding=F.relu(embedding)
        embedding=self.hidden_layers(embedding)
        output=self.output_layer(embedding)
        
        return output, None, (None, None)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0., batch_first=False):
        super().__init__()
        self.dropout=nn.Dropout(p=dropout)
        pe_shape=(1,seq_len,d_model) if batch_first else (seq_len,1,d_model)
        self.pe=nn.Parameter(torch.randn(*pe_shape))
    
    def forward(self, x):
        x=x+self.pe
        return self.dropout(x)

class PositionalEncoding1D(nn.Module):
     def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding1D,self).__init__()
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)
     
     def forward(self, x):
        x=x+self.pe[:x.size(0),:]
        return x

class ExtroceptiveEncoder(nn.Module):
    def __init__(self, obs_dim):
        super(ExtroceptiveEncoder, self).__init__()
        hidden_dims=cfg.MODEL.EXTROCEPTIVE_ENCODER.get('HIDDEN_DIMS',[64])
        mlp_dims=[obs_dim]+hidden_dims
        self.encoder=tu.make_mlp_default(mlp_dims)
        self.obs_feat_dim=mlp_dims[-1]
        self.keys_to_encode=None
    
    def forward(self, obs):    
        return self.encoder(obs)


class SWATPEEncoder(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe_dim = [d_model // len(cfg.MODEL.TRANSFORMER.TRAVERSALS) for _ in cfg.MODEL.TRANSFORMER.TRAVERSALS]
        self.pe_dim[-1] = d_model - self.pe_dim[0] * (len(cfg.MODEL.TRANSFORMER.TRAVERSALS) - 1)
        print (self.pe_dim)
        self.swat_pe = nn.ModuleList([nn.Embedding(seq_len, dim) for dim in self.pe_dim])

class TransformerModel(nn.Module):
    def __init__(self, obs_space, decoder_out_dim):
        super(TransformerModel, self).__init__()

        # --- Basic Setup & Validation ---
        self.decoder_out_dim = decoder_out_dim
        self.model_args = cfg.MODEL.TRANSFORMER
        self.seq_len = cfg.MODEL.MAX_LIMBS
        if not isinstance(obs_space, (spaces.Dict, OrderedDict)):
            raise ValueError("Transformer expects Dict obs_space")
        required_keys = ["proprioceptive", "context", "obs_padding_mask"]
        for key in required_keys:
             if key not in obs_space.spaces: 
                raise ValueError(f"Missing key '{key}'")
        flat_proprio_dim=obs_space["proprioceptive"].shape[0]
        flat_context_dim=obs_space["context"].shape[0]
        if flat_proprio_dim%self.seq_len!=0:
            raise ValueError("Proprio dim !div by MAX_LIMBS")
        if flat_context_dim%self.seq_len!=0:
            raise ValueError("Context dim !div by MAX_LIMBS")
        self.limb_obs_size=flat_proprio_dim//self.seq_len
        self.context_obs_size=flat_context_dim//self.seq_len
        self.d_model=cfg.MODEL.LIMB_EMBED_SIZE

        self.per_node_embed = self.model_args.get('PER_NODE_EMBED', False)
        self.per_node_decoder = self.model_args.get('PER_NODE_DECODER', False)
        self.use_fixed_attention = self.model_args.get('FIX_ATTENTION', False)
        self.use_hypernet = self.model_args.get('HYPERNET', False)
        self.hn_embed = self.use_hypernet and self.model_args.get('HN_EMBED', False)
        self.hn_decoder = self.use_hypernet and self.model_args.get('HN_DECODER', False)
        self.context_decoder_fusion = self.model_args.get('CONTEXT_DECODER_FUSION', True)
        self.ext_feat_fusion = self.model_args.get('EXT_MIX', 'none')
        self.use_swat_pe = self.model_args.get('USE_SWAT_PE', False)
        self.use_swat_re = self.model_args.get('USE_SWAT_RE', False)

        # --- Add Assertions / Warnings for Incompatible Flags ---
        if self.model_args.get('CONSISTENT_DROPOUT', False):
            raise NotImplementedError("CONSISTENT_DROPOUT=True is not supported.")
        if self.per_node_decoder:
            raise NotImplementedError("PER_NODE_DECODER=True is not supported.")
        
        # Optional: Add warning if HN is used without context fusion, though it might be valid
        if self.hn_decoder and not self.context_decoder_fusion:
           print("Warning: HN Decoder enabled but CONTEXT_DECODER_FUSION is False.")

        # --- Embeddings ---
        if self.per_node_embed:
             # (Logic as before - depends on unimal_ids passed in forward)
             initrange=self.model_args.get('EMBED_INIT',0.1)
             num_morph_types=len(cfg.ENV.get('WALKERS',cfg.ROBOSUITE.get('TRAINING_MORPHOLOGIES',[1])))
             self.limb_embed_weights=nn.Parameter(torch.zeros(self.seq_len,num_morph_types,self.limb_obs_size,self.d_model).uniform_(-initrange,initrange))
             self.limb_embed_bias=nn.Parameter(torch.zeros(self.seq_len,num_morph_types,self.d_model))
             self.context_embed=nn.Linear(self.context_obs_size,self.d_model)
        else:
             self.limb_embed = nn.Linear(self.limb_obs_size, self.d_model)
             self.context_embed = nn.Linear(self.context_obs_size, self.d_model)

        # --- Positional Encoding ---
        pos_embedding_type = self.model_args.get('POS_EMBEDDING', 'learnt')
        if pos_embedding_type=="learnt":
            self.pos_embedding=PositionalEncoding(self.d_model, self.seq_len)
        elif pos_embedding_type=="abs":
            self.pos_embedding=PositionalEncoding1D(self.d_model, self.seq_len)
        else: 
            self.pos_embedding=None

        # --- Transformer Encoder ---
        nhead=self.model_args.get('NHEAD',2)
        dim_feedforward=self.model_args.get('DIM_FEEDFORWARD',512)
        dropout=self.model_args.get('DROPOUT',0.1)
        nlayers=self.model_args.get('NLAYERS',3)
        encoder_layers=TransformerEncoderLayerResidual(self.d_model, nhead, dim_feedforward, dropout)
        encoder_norm=nn.LayerNorm(self.d_model)
        self.transformer_encoder=TransformerEncoder(encoder_layers, nlayers, norm=encoder_norm)

        # --- Extroceptive Handling ---
        self.extroceptive_encoder=None
        self.extro_feat_dim=0
        extro_keys=[]
        if "hfield" in cfg.ENV.get('KEYS_TO_KEEP',[]):
             extro_keys.append("hfield")
        
        extro_keys.extend(cfg.ROBOSUITE.get("EXTERO_KEYS",[]))
        extro_keys=list(set(extro_keys))
        self.keys_to_encode_extro=[k for k in extro_keys if k in obs_space.spaces]
        if self.keys_to_encode_extro:
            extro_dim=sum(np.prod(obs_space.spaces[k].shape) for k in self.keys_to_encode_extro)
            if extro_dim>0:
                self.extroceptive_encoder=ExtroceptiveEncoder(extro_dim)
                self.extro_feat_dim=self.extroceptive_encoder.obs_feat_dim
                self.extroceptive_encoder.keys_to_encode = self.keys_to_encode_extro

        # --- Decoder Input Dim Calculation ---
        decoder_input_dim=self.d_model
        if self.context_decoder_fusion:
            decoder_input_dim+=self.d_model
        if self.ext_feat_fusion=="late" and self.extroceptive_encoder is not None:
            decoder_input_dim+=self.extro_feat_dim
        self.decoder_input_dim = decoder_input_dim

        # --- Decoder Init ---
        if not self.per_node_decoder: # Only init standard decoder if PER_NODE is false
             decoder_dims = self.model_args.get('DECODER_DIMS', [128])
             self.decoder = tu.make_mlp_default([self.decoder_input_dim] + decoder_dims + [decoder_out_dim], final_nonlinearity=False)

        # --- FA/HN Component Init ---
        if self.use_fixed_attention:
            self.context_embed_attention=nn.Linear(self.context_obs_size,self.model_args.CONTEXT_EMBED_SIZE)
            context_encoder_layer=TransformerEncoderLayerResidual(self.model_args.CONTEXT_EMBED_SIZE,self.model_args.NHEAD,self.model_args.DIM_FEEDFORWARD,self.model_args.DROPOUT)
            self.context_encoder_attention=TransformerEncoder(context_encoder_layer,self.model_args.CONTEXT_LAYER,norm=None)
            # skip for for robosuite
            if self.model_args.get('HFIELD_IN_FIX_ATTENTION') and "hfield" in obs_space.spaces:
                self.context_hfield_encoder=ExtroceptiveEncoder(obs_space.spaces["hfield"].shape[0])
                self.context_compress=nn.Sequential(nn.Linear(self.context_hfield_encoder.obs_feat_dim+self.model_args.CONTEXT_EMBED_SIZE,self.model_args.CONTEXT_EMBED_SIZE),nn.ReLU())
            
        if self.use_hypernet:
            self.context_embed_HN=nn.Linear(self.context_obs_size,self.model_args.CONTEXT_EMBED_SIZE)
            modules=[nn.ReLU()]
            [modules.extend([nn.Linear(self.model_args.CONTEXT_EMBED_SIZE,self.model_args.CONTEXT_EMBED_SIZE),nn.ReLU()]) for _ in range(self.model_args.HN_CONTEXT_LAYER_NUM)]
            self.context_encoder_HN=nn.Sequential(*modules)
            HN_input_dim=self.model_args.CONTEXT_EMBED_SIZE
        if self.hn_embed:
            self.hnet_embed_weight=nn.Linear(HN_input_dim,self.limb_obs_size*self.d_model)
            self.hnet_embed_bias=nn.Linear(HN_input_dim,self.d_model)
        if self.hn_decoder:
            self.decoder_dims_hn=[self.decoder_input_dim]+self.model_args.DECODER_DIMS+[decoder_out_dim]
            self.hnet_decoder_weight=[]
            self.hnet_decoder_bias=[]
            [self.hnet_decoder_weight.append(nn.Linear(HN_input_dim,self.decoder_dims_hn[i]*self.decoder_dims_hn[i+1])) for i in range(len(self.decoder_dims_hn)-1)]
            [self.hnet_decoder_bias.append(nn.Linear(HN_input_dim,self.decoder_dims_hn[i+1])) for i in range(len(self.decoder_dims_hn)-1)]
            self.hnet_decoder_weight=nn.ModuleList(self.hnet_decoder_weight)
            self.hnet_decoder_bias=nn.ModuleList(self.hnet_decoder_bias)

        # --- SWAT PE Init ---
        if self.use_swat_pe: 
            self.swat_PE_encoder = SWATPEEncoder(self.d_model, self.seq_len)

        self.dropout = nn.Dropout(p=self.model_args.get('DROPOUT', 0.1))
        self.init_weights() # Call init

    def init_weights(self):
        # (Implementation remains the same - initializes existing layers)
        initrange = self.model_args.get('EMBED_INIT', 0.1)
        if not self.per_node_embed and hasattr(self, 'limb_embed'):
             self.limb_embed.weight.data.uniform_(-initrange,initrange)
             self.limb_embed.bias.data.zero_()
        if hasattr(self, 'context_embed'):
             self.context_embed.weight.data.uniform_(-initrange,initrange)
             self.context_embed.bias.data.zero_()
        if not self.per_node_decoder and hasattr(self, 'decoder'):
             initrange_dec = self.model_args.get('DECODER_INIT', 0.01)
             dec_last = self.decoder[-1] if isinstance(self.decoder, nn.Sequential) else self.decoder
             if isinstance(dec_last, nn.Linear):
                dec_last.bias.data.zero_()
                dec_last.weight.data.uniform_(-initrange_dec, initrange_dec)
        if self.use_fixed_attention and hasattr(self,'context_embed_attention'):
             initrange_fa=self.model_args.get('EMBED_INIT',0.1)
             self.context_embed_attention.weight.data.uniform_(-initrange_fa,initrange_fa)
        if self.use_hypernet and hasattr(self,'context_embed_HN'):
             initrange_hn=self.model_args.get('HN_EMBED_INIT',0.1)
             self.context_embed_HN.weight.data.uniform_(-initrange_hn,initrange_hn)
        if self.hn_embed:
             initrange_emb=self.model_args.get('EMBED_INIT',0.1)
             self.hnet_embed_weight.weight.data.zero_()
             self.hnet_embed_weight.bias.data.uniform_(-initrange_emb,initrange_emb)
             self.hnet_embed_bias.weight.data.zero_()
             self.hnet_embed_bias.bias.data.zero_()
        if self.hn_decoder:
             initrange_dec=self.model_args.get('DECODER_INIT',0.01)
             [ (w.weight.data.zero_(), w.bias.data.uniform_(-initrange_dec,initrange_dec)) for w in self.hnet_decoder_weight ]
             [ (b.weight.data.zero_(), b.bias.data.zero_()) for b in self.hnet_decoder_bias ]


    def forward(self, obs_proprio, obs_mask, obs_env, obs_context,
                morphology_info=None, return_attention=False, unimal_ids=None):
        # --- Handle PER_NODE_EMBED needing unimal_ids ---
        if self.per_node_embed and unimal_ids is None:
             # If IDs not provided (e.g., during Robosuite run), use zeros as fallback
             # This assumes weights for index 0 are generic or suitable defaults
             print("Warning: PER_NODE_EMBED=True but unimal_ids not provided. Using index 0.")
             batch_size_ids = obs_proprio.shape[0]
             unimal_ids = torch.zeros(batch_size_ids, dtype=torch.long, device=obs_proprio.device)

        # (Rest of forward implementation remains the same as previous version)
        # ... Reshape inputs ...
        batch_size = obs_proprio.shape[0]
        obs_proprio_r = obs_proprio.view(batch_size, self.seq_len, self.limb_obs_size)
        obs_context_r = obs_context.view(batch_size, self.seq_len, self.context_obs_size)
        obs_proprio_t = obs_proprio_r.permute(1, 0, 2)
        obs_context_t = obs_context_r.permute(1, 0, 2)

        # ... Context Processing ...
        context_embedding_fa = None
        context_embedding_hn = None
        context_embedding_shared = None
        if self.use_fixed_attention or self.use_hypernet or self.context_decoder_fusion:
             context_embedding_shared = self.context_embed(obs_context_t)
             if self.use_fixed_attention:
                ctx_embed_fa=self.context_embed_attention(obs_context_t)
             # HField Fusion...; context_embedding_fa=self.context_encoder_attention(ctx_embed_fa)
             if self.use_hypernet: 
                ctx_embed_hn=self.context_embed_HN(obs_context_t)
                context_embedding_hn=self.context_encoder_HN(ctx_embed_hn)

        # ... Proprio Embedding ...
        if self.hn_embed:
            embed_w=self.hnet_embed_weight(context_embedding_hn).view(self.seq_len,batch_size,self.limb_obs_size,self.d_model)
            embed_b=self.hnet_embed_bias(context_embedding_hn)
            proprio_embed=(obs_proprio_t.unsqueeze(-1)*embed_w).sum(dim=-2)+embed_b
        elif self.per_node_embed:
            proprio_embed=(obs_proprio_t.unsqueeze(-1)*self.limb_embed_weights[:,unimal_ids]).sum(dim=-2)+self.limb_embed_bias[:,unimal_ids]
        else:
            proprio_embed = self.limb_embed(obs_proprio_t)

        transformer_input = proprio_embed
        if self.model_args.get('EMBEDDING_SCALE', True): transformer_input *= math.sqrt(self.d_model)
        if self.pos_embedding is not None: transformer_input = self.pos_embedding(transformer_input)
        # SWAT PE ...
        if self.use_swat_pe and morphology_info and 'traversals' in morphology_info:
            trav=morphology_info['traversals']
        if trav is not None:
            transformer_input=self.swat_PE_encoder(transformer_input,trav)
        if self.model_args.get('EMBEDDING_DROPOUT',True):
            transformer_input = self.dropout(transformer_input)

        src_key_padding_mask = obs_mask.bool() if obs_mask is not None else None
        attn_mask = morphology_info['SWAT_RE'] if self.use_swat_re and morphology_info and 'SWAT_RE' in morphology_info else None
        context_for_transformer_layers = context_embedding_fa if self.use_fixed_attention else None
        attention_maps = None
        if return_attention:
            obs_encoded, attention_maps = self.transformer_encoder.get_attention_maps(
                                                transformer_input,
                                                mask=attn_mask,
                                                src_key_padding_mask=src_key_padding_mask,
                                                context=context_for_transformer_layers,
                                                morphology_info=morphology_info,
                                            )
        else: obs_encoded = self.transformer_encoder(
                                                transformer_input,
                                                mask=attn_mask,
                                                src_key_padding_mask=src_key_padding_mask,
                                                context=context_for_transformer_layers,
                                                morphology_info=morphology_info
                                            )

        # ... Prepare Decoder Input ...
        decoder_input = obs_encoded
        if self.context_decoder_fusion and context_embedding_shared is not None:
            decoder_input = torch.cat([decoder_input, context_embedding_shared], dim=-1)
        if self.ext_feat_fusion == "late" and self.extroceptive_encoder is not None:
            extro_list = [obs_env[k].view(batch_size, -1) for k in self.keys_to_encode_extro if k in obs_env]
            if extro_list:
                extro_cat=torch.cat(extro_list, dim=1)
                extro_encoded=self.extroceptive_encoder(extro_cat)
                extro_embedding_late=extro_encoded.unsqueeze(0).repeat(self.seq_len, 1, 1)
            if decoder_input.shape[1] == extro_embedding_late.shape[1]:
                decoder_input = torch.cat([decoder_input, extro_embedding_late], dim=-1)

        if decoder_input.shape[-1] != self.decoder_input_dim:
             raise RuntimeError(f"Decoder input dim mismatch! Actual {decoder_input.shape[-1]} != Expected {self.decoder_input_dim}. Check fusion flags.")

        # ... Apply Decoder ...
        if self.hn_decoder: # Check hn_decoder flag directly
            output=decoder_input; layer_num=len(self.hnet_decoder_weight)
            for i in range(layer_num):
                layer_w=self.hnet_decoder_weight[i](context_embedding_hn).view(self.seq_len,batch_size,self.decoder_dims_hn[i],self.decoder_dims_hn[i+1])
                layer_b=self.hnet_decoder_bias[i](context_embedding_hn)
                output = torch.einsum('sbi,sbio->sbo', output, layer_w) + layer_b
                if i!=(layer_num-1):
                    output=F.relu(output)
        # elif self.per_node_decoder: # This is handled by the check in __init__ now
        #      pass
        else:
            output = self.decoder(decoder_input)

        # Reshape Output
        output = output.permute(1, 0, 2).reshape(batch_size, -1)
        return output, attention_maps, (None, None)


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super(ActorCritic, self).__init__(); self.seq_len=cfg.MODEL.MAX_LIMBS
        is_dict_obs=isinstance(obs_space,(spaces.Dict, OrderedDict))
    if cfg.MODEL.TYPE=='transformer':
        if not is_dict_obs:
            raise ValueError("Transformer needs Dict obs")
            self.v_net=TransformerModel(obs_space,1)
            action_dim_per_node=1
            if cfg.ENV_NAME=="Unimal-v0":
                action_dim_per_node=2
            self.mu_net=TransformerModel(obs_space,action_dim_per_node)
            self.num_actions_policy_output=self.seq_len*action_dim_per_node
    elif cfg.MODEL.TYPE=='mlp':
        if not is_dict_obs:
            raise ValueError("MLP needs Dict obs")
            base_action_dim=action_space.shape[0]
            self.v_net=MLPModel(obs_space,1); self.mu_net=MLPModel(obs_space,base_action_dim)
            self.num_actions_policy_output=base_action_dim
    else: 
        raise ValueError(f"Bad MODEL.TYPE: {cfg.MODEL.TYPE}")
    action_std_dim=self.num_actions_policy_output
    if cfg.MODEL.ACTION_STD_FIXED:
        log_std_val=np.log(cfg.MODEL.ACTION_STD)
        self.log_std=nn.Parameter(log_std_val*torch.ones(1,action_std_dim),requires_grad=False) # check this (potential issue)
    else:
        self.log_std = nn.Parameter(torch.zeros(1, action_std_dim))

    def forward(self, obs, act=None, return_attention=False, unimal_ids=None, compute_val=True):
        if isinstance(obs,dict):
            batch_size=next(iter(obs.values())).shape[0]
        elif isinstance(obs,torch.Tensor):
            batch_size=obs.shape[0]
        else:
            raise TypeError(f"Bad obs type: {type(obs)}")
        val=torch.zeros(batch_size,1,device=obs.get("proprioceptive",next(iter(obs.values()))).device)
        v_attention_maps,mu_attention_maps=None,None
        dv_out,dm_out=None,None
        if cfg.MODEL.TYPE=='transformer':
            obs_proprio=obs.get("proprioceptive")
            obs_context=obs.get("context")
            obs_mask=obs.get("obs_padding_mask")
            act_mask=obs.get("act_padding_mask")
            if obs_proprio is None or obs_mask is None or obs_context is None:
                raise ValueError("Transformer missing keys.")
            core_keys={"proprioceptive","context","edges","obs_padding_mask","act_padding_mask","traversals","SWAT_RE"}
            obs_env={k:v for k,v in obs.items() if k not in core_keys}
            morphology_info={k:v for k,v in obs.items() if k in ["traversals","SWAT_RE"]}
            morphology_info=morphology_info or None
            if compute_val:
                 raw_vals,v_attention_maps,(_,_)=self.v_net(obs_proprio=obs_proprio,obs_mask=obs_mask,obs_env=obs_env,obs_context=obs_context,morphology_info=morphology_info,return_attention=return_attention,unimal_ids=unimal_ids)
                 limb_vals=raw_vals.view(batch_size,self.v_net.seq_len,1)
                 mask_for_val=(~obs_mask.bool()).float().unsqueeze(-1)
                 if mask_for_val.shape[1]!=limb_vals.shape[1]:
                    mask_for_val=mask_for_val[:,:limb_vals.shape[1],:]
                 limb_vals=limb_vals*mask_for_val; num_limbs=torch.clamp(mask_for_val.sum(dim=1),min=1.0); val=torch.sum(limb_vals,dim=1)/num_limbs
            mu,mu_attention_maps,(_,_)=self.mu_net(obs_proprio=obs_proprio,obs_mask=obs_mask,obs_env=obs_env,obs_context=obs_context,morphology_info=morphology_info,return_attention=return_attention,unimal_ids=unimal_ids)
        elif cfg.MODEL.TYPE=='mlp':
            obs_proprio_flat=obs.get("proprioceptive")
            if obs_proprio_flat is None:
                raise ValueError("Missing 'proprioceptive' for MLP.")
            obs_env_mlp={k:v for k,v in obs.items() if k!="proprioceptive"}
            if compute_val:
                val,_,(_,_)=self.v_net(obs_proprio=obs_proprio_flat,obs_env=obs_env_mlp)
            mu,_,(_,_)=self.mu_net(obs_proprio=obs_proprio_flat,obs_env=obs_env_mlp)
        std=torch.exp(self.log_std)
        if std.shape[1]!=self.num_actions_policy_output:
            if std.shape==(1,1):
                std=std.repeat(1,self.num_actions_policy_output)
            else:
                raise ValueError(f"Std dim mismatch")
        if mu.shape[1]!=self.num_actions_policy_output:
            raise ValueError(f"Mu dim mismatch")
        pi=Normal(mu,std)
        if act is not None:
            if act.shape!=mu.shape:
                raise ValueError(f"Action shape {act.shape}!=mu shape {mu.shape}")
            current_act_mask=obs.get("act_padding_mask")
            action_mask_for_calc=torch.zeros_like(mu,dtype=torch.bool)
            if current_act_mask is not None:
                current_act_mask_bool=current_act_mask.bool()
                if mu.shape[1]==current_act_mask_bool.shape[1]:
                    action_mask_for_calc=current_act_mask_bool
                elif cfg.MODEL.TYPE=='transformer' and mu.shape[1]>current_act_mask_bool.shape[1]:
                    action_dim_per_node=mu.shape[1]//self.seq_len
                    if mu.shape[1]%self.seq_len==0 and current_act_mask_bool.shape[1]==self.seq_len:
                        action_mask_for_calc=current_act_mask_bool.unsqueeze(-1).repeat(1,1,action_dim_per_node).view(batch_size,-1)
                    else:
                        raise ValueError("Cannot broadcast act_mask")
                if action_mask_for_calc.shape!=mu.shape:
                    raise ValueError("Reshaped act_mask!=mu shape")
            logp=pi.log_prob(act); logp[action_mask_for_calc]=0.0; logp=logp.sum(-1,keepdim=True)
            entropy=pi.entropy(); entropy[action_mask_for_calc]=0.0; num_valid_actions=(~action_mask_for_calc).float().sum(dim=1,keepdim=True)
            num_valid_actions=torch.clamp(num_valid_actions,min=1.0)
            entropy=(entropy.sum(-1,keepdim=True)/num_valid_actions).mean()
            return val,pi,logp,entropy,None,None
        else:
            if return_attention:
                return val,pi,v_attention_maps,mu_attention_maps,None,None
            else:
                return val,pi,None,None,None,None


class Agent:
    def __init__(self, actor_critic):
        self.ac=actor_critic
    @torch.no_grad()
    def act(self, obs, return_attention=False, unimal_ids=None, compute_val=True):
        val,pi,v_attention_maps,mu_attention_maps,_,_=self.ac(obs,act=None,return_attention=return_attention,unimal_ids=unimal_ids,compute_val=compute_val);
        if not cfg.DETERMINISTIC:
            act_out=pi.sample()
        else:
            act_out=pi.loc
        logp=pi.log_prob(act_out)
        act_mask=obs.get("act_padding_mask",None)
        action_mask_for_calc=torch.zeros_like(logp,dtype=torch.bool)
        if act_mask is not None:
            act_mask_bool=act_mask.bool()
            if logp.shape[1]==act_mask_bool.shape[1]:
                action_mask_for_calc=act_mask_bool
            elif cfg.MODEL.TYPE=='transformer' and logp.shape[1]>act_mask_bool.shape[1]:
                 action_dim_per_node=logp.shape[1]//self.ac.seq_len
                 if logp.shape[1]%self.ac.seq_len==0 and act_mask_bool.shape[1]==self.ac.seq_len:
                    batch_size_act=act_mask.shape[0]
                 action_mask_for_calc=act_mask_bool.unsqueeze(-1).repeat(1,1,action_dim_per_node).view(batch_size_act,-1)
            if action_mask_for_calc.shape==logp.shape:
                logp[action_mask_for_calc]=0.0
        logp=logp.sum(-1,keepdim=True)
        return val,act_out,logp,None,None
    @torch.no_grad()
    def get_value(self, obs, unimal_ids=None):
        val,_,_,_,_,_=self.ac(obs,act=None,return_attention=False,unimal_ids=unimal_ids,compute_val=True)
        return val