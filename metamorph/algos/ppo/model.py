# metamorph/algos/ppo/model.py

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from collections import OrderedDict
from gym import spaces # Import spaces for type checking

from metamorph.config import cfg
from metamorph.utils import model as tu

# Assuming transformer definitions are in this relative path
from .transformer import TransformerEncoder
from .transformer import TransformerEncoderLayerResidual

# MLP model as single-robot baseline (assuming it's adapted for flattened input)
class MLPModel(nn.Module):
    def __init__(self, obs_space, out_dim):
        super(MLPModel, self).__init__()
        self.model_args = cfg.MODEL.MLP
        self.seq_len = cfg.MODEL.MAX_LIMBS # Needed for consistency, though MLP doesn't use seq structure explicitly

        # Verify obs_space structure for MLPFlattener output
        if not isinstance(obs_space, (spaces.Dict, OrderedDict)) or "proprioceptive" not in obs_space.spaces:
            raise ValueError("MLPModel requires a Dict obs_space with 'proprioceptive' key (from MLPFlattener).")

        flat_proprio_dim = obs_space["proprioceptive"].shape[0]
        # out_dim is 1 for critic, or action_dim for actor
        self.output_dim = out_dim

        self.input_layer = nn.Linear(flat_proprio_dim, self.model_args.HIDDEN_DIM)
        self.output_layer = nn.Linear(self.model_args.HIDDEN_DIM, out_dim)

        # --- Handle potential hfield input ---
        # Note: This assumes hfield would be concatenated *before* flattening,
        # which might not be how MLPFlattener works. This section might need
        # removal or adaptation depending on how external features are passed to MLP.
        # For now, assume hfield is NOT part of the MLP pipeline by default.
        hidden_input_dim = self.model_args.HIDDEN_DIM
        self.hfield_encoder = None
        # if "hfield" in cfg.ENV.KEYS_TO_KEEP and "hfield" in obs_space.spaces:
        #     # This likely won't trigger if MLPFlattener defines the space
        #     print("MLP: Initializing HField Encoder")
        #     self.hfield_encoder = HFieldObsEncoder(obs_space.spaces["hfield"].shape[0])
        #     hidden_input_dim += self.hfield_encoder.obs_feat_dim

        hidden_dims = [hidden_input_dim] + [self.model_args.HIDDEN_DIM for _ in range(self.model_args.LAYER_NUM - 1)]
        self.hidden_layers = tu.make_mlp_default(hidden_dims)

    # MLP forward expects the specific flattened 'proprioceptive' key from the obs dict
    def forward(self, obs_proprio, obs_env=None, # Simplified signature for MLP path
                return_attention=False, dropout_mask=None, unimal_ids=None, **kwargs): # Use kwargs to ignore unused args

        embedding = self.input_layer(obs_proprio) # Input is the flat tensor
        embedding = F.relu(embedding)

        # Incorporate hfield if it exists (assuming obs_env is passed)
        # if self.hfield_encoder is not None and obs_env and "hfield" in obs_env:
        #     hfield_embedding = self.hfield_encoder(obs_env["hfield"])
        #     embedding = torch.cat([embedding, hfield_embedding], 1)

        embedding = self.hidden_layers(embedding)
        output = self.output_layer(embedding)

        # MLP returns: output, None (no attention), Tuple of None (no specific dropout masks)
        return output, None, (None, None)


# --- Positional Encodings ---
class PositionalEncoding(nn.Module):
    # Standard learned positional embedding
    def __init__(self, d_model, seq_len, dropout=0., batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Parameter shape depends on batch_first convention (Transformer uses batch_first=False)
        pe_shape = (1, seq_len, d_model) if batch_first else (seq_len, 1, d_model)
        self.pe = nn.Parameter(torch.randn(*pe_shape))

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class PositionalEncoding1D(nn.Module):
    # Sinusoidal encoding (from original Transformer paper)
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding1D, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

# --- Observation Encoders ---
class HFieldObsEncoder(nn.Module):
    """Encoder for hfield observation."""
    def __init__(self, obs_dim):
        super(HFieldObsEncoder, self).__init__()
        # Use configured dims or provide a default
        hidden_dims = cfg.MODEL.TRANSFORMER.get('EXT_HIDDEN_DIMS', [64]) # Safer access
        mlp_dims = [obs_dim] + hidden_dims
        self.encoder = tu.make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)

class ExtroceptiveEncoder(nn.Module):
    """Encoder for general extroceptive features (e.g., object state, eef pose)."""
    def __init__(self, obs_dim):
        super(ExtroceptiveEncoder, self).__init__()
        # Use configured dims or provide a default
        hidden_dims = cfg.MODEL.EXTROCEPTIVE_ENCODER.get('HIDDEN_DIMS', [64]) # Safer access
        mlp_dims = [obs_dim] + hidden_dims
        self.encoder = tu.make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)

# --- Transformer Model ---
class TransformerModel(nn.Module):
    def __init__(self, obs_space, decoder_out_dim):
        super(TransformerModel, self).__init__()

        if not isinstance(obs_space, (spaces.Dict, OrderedDict)):
             raise ValueError(f"TransformerModel expects a Dict obs_space, got {type(obs_space)}")

        self.decoder_out_dim = decoder_out_dim
        self.model_args = cfg.MODEL.TRANSFORMER
        self.seq_len = cfg.MODEL.MAX_LIMBS

        # --- Determine Input Feature Sizes ---
        if "proprioceptive" not in obs_space.spaces or "context" not in obs_space.spaces:
             raise ValueError("Transformer requires 'proprioceptive' and 'context' keys.")
        # Assuming flattened inputs (Batch, Seq*Feat) from wrappers
        self.limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.context_obs_size = obs_space["context"].shape[0] // self.seq_len

        self.d_model = cfg.MODEL.LIMB_EMBED_SIZE

        # --- Input Embeddings ---
        # Currently assumes PER_NODE_EMBED is False for simplicity
        if self.model_args.get('PER_NODE_EMBED', False):
             raise NotImplementedError("PER_NODE_EMBED not fully integrated/verified.")
        self.limb_embed = nn.Linear(self.limb_obs_size, self.d_model)
        self.context_embed = nn.Linear(self.context_obs_size, self.d_model) # For context features

        self.ext_feat_fusion = self.model_args.get('EXT_MIX', 'none') # Default to none

        # --- Positional Encoding ---
        pos_embedding_type = self.model_args.get('POS_EMBEDDING', 'learnt') # Default to learnt
        if pos_embedding_type == "learnt":
            print ('Using learned PE')
            self.pos_embedding = PositionalEncoding(self.d_model, self.seq_len, batch_first=False)
        elif pos_embedding_type == "abs":
            print ('Using absolute (sinusoidal) PE')
            self.pos_embedding = PositionalEncoding1D(self.d_model, self.seq_len)
        else:
            print ('No PE used.')
            self.pos_embedding = None

        # --- Transformer Encoder ---
        # Use .get() for safer access to config values
        nhead = self.model_args.get('NHEAD', 2)
        dim_feedforward = self.model_args.get('DIM_FEEDFORWARD', 512)
        dropout = self.model_args.get('DROPOUT', 0.1)
        nlayers = self.model_args.get('NLAYERS', 3)

        encoder_layers = TransformerEncoderLayerResidual(
            self.d_model, nhead, dim_feedforward, dropout, batch_first=False
        )
        # Apply layer norm after encoder stack
        encoder_norm = nn.LayerNorm(self.d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, norm=encoder_norm)

        # --- H-Field Encoder (Conditional) ---
        self.hfield_encoder = None
        if "hfield" in cfg.ENV.KEYS_TO_KEEP and "hfield" in obs_space.spaces:
            print ('Initializing HField Encoder')
            self.hfield_encoder = HFieldObsEncoder(obs_space.spaces["hfield"].shape[0])

        # --- Robosuite Extroceptive Encoder (Conditional) ---
        self.extroceptive_encoder = None
        self.extro_feat_dim = 0
        self.extroceptive_keys = [k for k in ['object-state', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel'] if k in obs_space.spaces]
        if self.extroceptive_keys:
            extroceptive_dim = sum(np.prod(obs_space.spaces[k].shape) for k in self.extroceptive_keys) # Use prod for multi-dim shapes
            print(f"Initializing Extroceptive Encoder for keys: {self.extroceptive_keys}, total dim: {extroceptive_dim}")
            self.extroceptive_encoder = ExtroceptiveEncoder(extroceptive_dim)
            self.extro_feat_dim = self.extroceptive_encoder.obs_feat_dim

        # --- Calculate Decoder Input Dimension ---
        decoder_input_dim = self.d_model
        if self.ext_feat_fusion == "late":
            if self.hfield_encoder is not None:
                decoder_input_dim += self.hfield_encoder.obs_feat_dim
            if self.extroceptive_encoder is not None:
                decoder_input_dim += self.extro_feat_dim
        print(f"Decoder input dimension: {decoder_input_dim}")
        self.decoder_input_dim = decoder_input_dim # Store for potential use (e.g., HN)

        # --- Decoder ---
        if self.model_args.get('PER_NODE_DECODER', False):
            raise NotImplementedError("PER_NODE_DECODER not fully integrated/verified.")
        else:
            decoder_dims = self.model_args.get('DECODER_DIMS', [128]) # Example default
            self.decoder = tu.make_mlp_default(
                [self.decoder_input_dim] + decoder_dims + [decoder_out_dim],
                final_nonlinearity=False,
            )

        # --- Fixed Attention / HyperNet Stubs (Conditional) ---
        self.context_embed_attention = None
        self.context_encoder_attention = None
        if self.model_args.get('FIX_ATTENTION', False):
             print ('Initializing Fixed Attention components (stub)')
             context_embed_size = self.model_args.get('CONTEXT_EMBED_SIZE', 128)
             self.context_embed_attention = nn.Linear(self.context_obs_size, context_embed_size)
             # Example simple encoder
             self.context_encoder_attention = nn.Sequential(nn.Linear(context_embed_size, context_embed_size), nn.ReLU())
        # Add HYPERNET init here if needed, similar structure

        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()


    def init_weights(self):
        # Simplified init
        initrange = self.model_args.get('EMBED_INIT', 0.1)
        self.limb_embed.weight.data.uniform_(-initrange, initrange)
        if self.limb_embed.bias is not None: self.limb_embed.bias.data.zero_()
        self.context_embed.weight.data.uniform_(-initrange, initrange)
        if self.context_embed.bias is not None: self.context_embed.bias.data.zero_()

        if hasattr(self, 'decoder'): # PER_NODE_DECODER=False path
             initrange = self.model_args.get('DECODER_INIT', 0.01)
             # Init last layer of decoder MLP
             if isinstance(self.decoder, nn.Sequential) and len(self.decoder) > 0 and isinstance(self.decoder[-1], nn.Linear):
                  self.decoder[-1].bias.data.zero_()
                  self.decoder[-1].weight.data.uniform_(-initrange, initrange)
             elif isinstance(self.decoder, nn.Linear): # If decoder is just one layer
                  self.decoder.bias.data.zero_()
                  self.decoder.weight.data.uniform_(-initrange, initrange)

        if self.context_embed_attention:
            initrange = self.model_args.get('EMBED_INIT', 0.1)
            self.context_embed_attention.weight.data.uniform_(-initrange, initrange)
        # Add init for context_encoder_attention, HyperNet components if implemented


    # --- Updated forward signature ---
    def forward(self, obs_proprio, obs_mask, obs_env, obs_context, morphology_info,
                return_attention=False, dropout_mask=None, unimal_ids=None):

        # Inputs obs_proprio, obs_context are expected to be (Batch, Seq*Feat)
        batch_size = obs_proprio.shape[0]

        # Reshape flattened inputs: (Batch, Seq*Feat) -> (Batch, Seq, Feat)
        obs_proprio_r = obs_proprio.view(batch_size, self.seq_len, self.limb_obs_size)
        obs_context_r = obs_context.view(batch_size, self.seq_len, self.context_obs_size)

        # Transformer expects (Seq, Batch, Feat)
        obs_proprio_t = obs_proprio_r.permute(1, 0, 2)
        obs_context_t = obs_context_r.permute(1, 0, 2)

        # --- Input Embedding & PE ---
        obs_embed = self.limb_embed(obs_proprio_t) # (Seq, Batch, d_model)
        if self.model_args.get('EMBEDDING_SCALE', True):
            obs_embed *= math.sqrt(self.d_model)
        if self.pos_embedding is not None:
            obs_embed = self.pos_embedding(obs_embed)

        # --- H-field and Extroceptive Processing ---
        hfield_embedding_late = None
        if self.hfield_encoder is not None and obs_env and "hfield" in obs_env:
            hfield_obs = self.hfield_encoder(obs_env["hfield"]) # (Batch, hfield_feat_dim)
            if self.ext_feat_fusion == "late":
                hfield_embedding_late = hfield_obs.unsqueeze(0).repeat(self.seq_len, 1, 1) # (Seq, Batch, hfield_feat_dim)

        extroceptive_embedding_late = None
        if self.extroceptive_encoder is not None and self.extroceptive_keys:
            extro_list = [obs_env[k].view(batch_size, -1) for k in self.extroceptive_keys if k in obs_env]
            if extro_list:
                 extro_cat = torch.cat(extro_list, dim=1)
                 extro_encoded = self.extroceptive_encoder(extro_cat) # (Batch, extro_feat_dim)
                 if self.ext_feat_fusion == "late":
                     extroceptive_embedding_late = extro_encoded.unsqueeze(0).repeat(self.seq_len, 1, 1) # (Seq, Batch, extro_feat_dim)

        # --- Apply Dropout ---
        if self.model_args.get('EMBEDDING_DROPOUT', True):
             obs_embed = self.dropout(obs_embed)

        # --- Context for FA/HN ---
        context_for_attention = None
        if self.model_args.get('FIX_ATTENTION', False) and self.context_encoder_attention is not None:
             context_embed = self.context_embed_attention(obs_context_t) # Use dedicated context embed
             context_for_attention = self.context_encoder_attention(context_embed)
        # Add HN logic here if implemented

        # --- Transformer Encoder ---
        # Mask should be (Batch, Seq)
        src_key_padding_mask = obs_mask.bool() if obs_mask is not None else None

        if return_attention:
             obs_encoded, attention_maps = self.transformer_encoder.get_attention_maps(
                 obs_embed, src_key_padding_mask=src_key_padding_mask, context=context_for_attention)
        else:
             attention_maps = None
             obs_encoded = self.transformer_encoder(
                 obs_embed, src_key_padding_mask=src_key_padding_mask, context=context_for_attention)

        # --- Late Fusion & Decoder ---
        decoder_input = obs_encoded
        if self.ext_feat_fusion == "late":
             if hfield_embedding_late is not None:
                  decoder_input = torch.cat([decoder_input, hfield_embedding_late], dim=-1)
             if extroceptive_embedding_late is not None:
                  decoder_input = torch.cat([decoder_input, extroceptive_embedding_late], dim=-1)

        output = self.decoder(decoder_input) # (Seq, Batch, decoder_out_dim)

        # --- Reshape Output ---
        output = output.permute(1, 0, 2) # (Batch, Seq, decoder_out_dim)
        output = output.reshape(batch_size, -1) # (Batch, Seq * decoder_out_dim)

        # --- Return dummy dropout masks (as tuple) ---
        # These placeholders are needed to match ActorCritic unpacking logic
        dummy_dropout_mask_v = torch.ones(batch_size, 12, 128, device=output.device) # Shape from buffer init
        dummy_dropout_mask_mu = torch.ones(batch_size, 12, 128, device=output.device)

        return output, attention_maps, (dummy_dropout_mask_v, dummy_dropout_mask_mu)


# === ActorCritic ===
class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super(ActorCritic, self).__init__()
        self.seq_len = cfg.MODEL.MAX_LIMBS
        is_dict_obs = isinstance(obs_space, (spaces.Dict, OrderedDict))

        # --- Model Selection ---
        if cfg.MODEL.TYPE == 'transformer':
            if not is_dict_obs: raise ValueError("Transformer model requires Dict observation space.")
            print("[ActorCritic] Using TransformerModel")
            self.v_net = TransformerModel(obs_space, 1) # Critic output dim = 1

            # Determine action dim per node based on env
            action_dim_per_node = 1 # Default for Robosuite/Modular
            if cfg.ENV_NAME == "Unimal-v0": action_dim_per_node = 2

            self.mu_net = TransformerModel(obs_space, action_dim_per_node)
            self.num_actions_total_padded = self.seq_len * action_dim_per_node

        elif cfg.MODEL.TYPE == 'mlp':
             if not is_dict_obs: raise ValueError("MLP model requires Dict observation space.")
             print("[ActorCritic] Using MLPModel")
             base_action_dim = action_space.shape[0]
             self.v_net = MLPModel(obs_space, 1)
             self.mu_net = MLPModel(obs_space, base_action_dim)
             self.num_actions_total_padded = base_action_dim
        else:
             raise ValueError(f"Unsupported MODEL.TYPE: {cfg.MODEL.TYPE}")

        # --- Action STD ---
        action_std_dim = self.num_actions_total_padded
        print(f"[ActorCritic] Action std dimension set to: {action_std_dim}")

        if cfg.MODEL.ACTION_STD_FIXED:
             log_std_val = np.log(cfg.MODEL.ACTION_STD)
             self.log_std = nn.Parameter(log_std_val * torch.ones(1, action_std_dim), requires_grad=False)
        else:
             self.log_std = nn.Parameter(torch.zeros(1, action_std_dim))

    def forward(self, obs, act=None, return_attention=False, dropout_mask_v=None, dropout_mask_mu=None, unimal_ids=None, compute_val=True):

        batch_size = next(iter(obs.values())).shape[0] # Get batch size from first obs tensor

        # Initialize outputs
        val = torch.zeros(batch_size, 1, device=next(iter(obs.values())).device)
        v_attention_maps, mu_attention_maps = None, None
        # Initialize dropout masks using the input arguments as defaults
        dropout_mask_v_out = dropout_mask_v
        dropout_mask_mu_out = dropout_mask_mu

        # --- Prepare inputs based on model type ---
        if cfg.MODEL.TYPE == 'transformer':
            # Extract Tensors, using .get() for safety
            obs_proprio = obs.get("proprioceptive")
            obs_context = obs.get("context")
            obs_mask = obs.get("obs_padding_mask")
            act_mask = obs.get("act_padding_mask")

            if obs_proprio is None or obs_context is None or obs_mask is None:
                raise ValueError("Missing required observation keys for Transformer.")
            if act_mask is None: # Needed later for masking
                raise ValueError("Missing 'act_padding_mask' key for Transformer.")

            # Prepare obs_env dict from non-core keys
            core_keys_t = {"proprioceptive", "context", "edges", "obs_padding_mask", "act_padding_mask"}
            obs_env_t = {k: v for k, v in obs.items() if k not in core_keys_t}

            # Prepare morphology info (minimal example)
            morphology_info_t = None # Set to None if no relevant keys found

            obs_mask_t = obs_mask.bool()
            act_mask_t = act_mask.bool()

            # --- Critic Path ---
            if compute_val:
                 # Call v_net (TransformerModel expects specific args)
                 raw_limb_vals, v_attention_maps, (dropout_mask_v_out, dropout_mask_mu_out) = self.v_net(
                     obs_proprio=obs_proprio, obs_mask=obs_mask_t, obs_env=obs_env_t,
                     obs_context=obs_context, morphology_info=morphology_info_t,
                     return_attention=return_attention, dropout_mask=dropout_mask_v, unimal_ids=unimal_ids)
                 # Aggregate value
                 limb_vals = raw_limb_vals.view(batch_size, self.seq_len, 1)
                 mask_for_val = (~obs_mask_t).float().unsqueeze(-1) # Invert mask: True=Valid
                 limb_vals = limb_vals * mask_for_val
                 num_limbs = torch.clamp(mask_for_val.sum(dim=1), min=1.0)
                 val = torch.sum(limb_vals, dim=1) / num_limbs # (Batch, 1)

            # --- Actor Path ---
            # Call mu_net, potentially overwriting dropout masks if compute_val was False
            mu, mu_attention_maps, (current_dropout_v, current_dropout_mu) = self.mu_net(
                 obs_proprio=obs_proprio, obs_mask=obs_mask_t, obs_env=obs_env_t,
                 obs_context=obs_context, morphology_info=morphology_info_t,
                 return_attention=return_attention, dropout_mask=dropout_mask_mu, unimal_ids=unimal_ids)
            # Update masks only if val wasn't computed (otherwise keep masks from v_net)
            if not compute_val:
                 dropout_mask_v_out = current_dropout_v
                 dropout_mask_mu_out = current_dropout_mu

        elif cfg.MODEL.TYPE == 'mlp':
            # --- MLP Path ---
            obs_proprio_flat = obs.get("proprioceptive")
            if obs_proprio_flat is None: raise ValueError("Missing 'proprioceptive' for MLP.")
            # MLP might not use obs_env, pass empty dict if needed
            obs_env_mlp = {}

            if compute_val:
                 val, _, (dropout_mask_v_out, dropout_mask_mu_out) = self.v_net(obs_proprio=obs_proprio_flat, obs_env=obs_env_mlp)
            # else: val remains zeros, masks keep input values

            mu, _, (current_dropout_v, current_dropout_mu) = self.mu_net(obs_proprio=obs_proprio_flat, obs_env=obs_env_mlp)
            if not compute_val: # Update masks if val wasn't computed
                 dropout_mask_v_out = current_dropout_v
                 dropout_mask_mu_out = current_dropout_mu
        else:
            raise ValueError(f"Unsupported MODEL.TYPE: {cfg.MODEL.TYPE}")

        # --- Action Distribution ---
        std = torch.exp(self.log_std)
        # Basic shape check and attempt to broadcast std
        if mu.shape[1] != std.shape[1]:
            if std.shape[0] == 1 and std.shape[1] < mu.shape[1]:
                 print(f"Warning: std dim {std.shape[1]} < mu dim {mu.shape[1]}. Padding std.")
                 std_padded = torch.zeros_like(mu[0:1,:]) # Shape (1, mu_dim)
                 std_padded[:, :std.shape[1]] = std
                 std = std_padded
            elif std.shape[0]==1 and std.shape[1] > mu.shape[1]:
                 print(f"Warning: std dim {std.shape[1]} > mu dim {mu.shape[1]}. Truncating std.")
                 std = std[:, :mu.shape[1]]
            else:
                 raise ValueError(f"Unresolvable shape mismatch: mu {mu.shape}, std {std.shape}")

        pi = Normal(mu, std)

        # --- Calculate Log Prob and Entropy if Action is Provided ---
        if act is not None:
            if act.shape != mu.shape: raise ValueError(f"Action shape {act.shape} != mu shape {mu.shape}.")
            logp = pi.log_prob(act)

            # Masking logic needs the correct mask based on model type
            if cfg.MODEL.TYPE == 'transformer':
                 current_act_mask = act_mask_t # (Batch, Seq) boolean mask
                 action_dim_per_node = self.mu_net.decoder_out_dim
                 # Repeat mask if multi-dim action per node
                 if action_dim_per_node > 1:
                      current_act_mask = current_act_mask.unsqueeze(-1).repeat(1, 1, action_dim_per_node).view(batch_size, -1)
            else: # MLP
                 current_act_mask = torch.zeros_like(logp, dtype=torch.bool) # No masking needed

            logp[current_act_mask] = 0.0 # Apply mask (True means mask out)
            logp = logp.sum(-1, keepdim=True)

            entropy = pi.entropy()
            entropy[current_act_mask] = 0.0 # Apply mask
            # Average entropy over valid dimensions
            num_valid_actions = (~current_act_mask).float().sum(dim=1, keepdim=True)
            num_valid_actions = torch.clamp(num_valid_actions, min=1.0)
            entropy = (entropy.sum(-1, keepdim=True) / num_valid_actions).mean() # Mean over batch

            return val, pi, logp, entropy, dropout_mask_v_out, dropout_mask_mu_out
        else:
            # Return values when no action is provided (e.g., for agent.act)
            if return_attention:
                return val, pi, v_attention_maps, mu_attention_maps, dropout_mask_v_out, dropout_mask_mu_out
            else:
                return val, pi, None, None, dropout_mask_v_out, dropout_mask_mu_out

# === Agent ===
class Agent:
    def __init__(self, actor_critic):
        self.ac = actor_critic

    @torch.no_grad()
    def act(self, obs, return_attention=False, dropout_mask_v=None, dropout_mask_mu=None, unimal_ids=None, compute_val=True):
        # Get outputs from ActorCritic, passing act=None
        val, pi, v_attention_maps, mu_attention_maps, dropout_mask_v_out, dropout_mask_mu_out = self.ac(
            obs, act=None, return_attention=return_attention,
            dropout_mask_v=dropout_mask_v, dropout_mask_mu=dropout_mask_mu,
            unimal_ids=unimal_ids, compute_val=compute_val)

        self.pi = pi # Store distribution if needed elsewhere

        # Sample or take mean action
        if not cfg.DETERMINISTIC:
            act_out = pi.sample()
        else:
            act_out = pi.loc

        # Calculate logp of the action *taken*
        logp = pi.log_prob(act_out)
        act_mask = obs.get("act_padding_mask", None) # Get mask from observation

        if act_mask is not None:
            act_mask_bool = act_mask.bool()
            # Reshape mask if needed (only for Transformer multi-dim action per node)
            if cfg.MODEL.TYPE == 'transformer':
                action_dim_per_node = getattr(self.ac.mu_net, 'decoder_out_dim', 1)
                if action_dim_per_node > 1:
                     batch_size = act_mask_bool.shape[0]
                     act_mask_bool = act_mask_bool.unsqueeze(-1).repeat(1, 1, action_dim_per_node).view(batch_size, -1)

            logp[act_mask_bool] = 0.0 # Apply mask (True means mask out)
        # else: Assume no mask needed (e.g., MLP)

        logp = logp.sum(-1, keepdim=True) # Sum log probabilities across action dimension

        # Return action (potentially padded) and masked logp
        return val, act_out, logp, dropout_mask_v_out, dropout_mask_mu_out


    @torch.no_grad()
    def get_value(self, obs, dropout_mask_v=None, dropout_mask_mu=None, unimal_ids=None):
        # Ensure compute_val=True when calling ActorCritic
        val, _, _, _, _, _ = self.ac(
            obs, act=None, return_attention=False,
            dropout_mask_v=dropout_mask_v, dropout_mask_mu=dropout_mask_mu,
            unimal_ids=unimal_ids, compute_val=True)
        return val