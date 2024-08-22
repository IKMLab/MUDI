from typing import Optional

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_utils import SequenceSummary
from transformers.models.bart.modeling_bart import (
    BART_ATTENTION_CLASSES,
    BartDecoderLayer,
)

from src.models.dgat_conv_finetuning import (
    DGatForCoherenceAwareDialogueEncoding,
    get_layer_class,
)
from src.modules.embedder import SoftPromptEmbedder
from src.utils.constants import (
    COHERENCE_RELATIONS,
    NEW_SPEICAL_TOKENS,
    NEW_SPEICAL_TOKENS_MAP,
)
from src.utils.model_configs import PersonalizedDialogueGeneratorConfig
from src.utils.model_outputs import (
    PersonalizedDialogueGeneratorOutput,
    PersonalizedDialogueGeneratorTrainerOutput,
)

MAX_MODEL_SEQ_LENGTH = 512  # or 1024 (default BART-large length)


class CoherenceAwareBartDecoderLayer(BartDecoderLayer):

    def __init__(self, config):
        super().__init__(config)

        self.coherence_attn_strategy = config.coherence_attn_strategy
        if config.coherence_attn_strategy in ('Emb', 'SP+Emb'):
            self.coherence_embeddings = nn.Embedding(len(COHERENCE_RELATIONS),
                                                     config.d_model)
        self.coherence_rels_indices = None
        self.coherence_rels_embs = None
        self.coherence_attn = BART_ATTENTION_CLASSES[
            config._attn_implementation](
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                config=config,
            )
        self.coherence_attn_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:
                                                  2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[
                2:4] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states,
                                                  p=self.dropout,
                                                  training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Coherence-aware Attention
        coherence_attn_present_key_value = None
        coherence_attn_weights = None

        residual = hidden_states
        coherence_attn_past_key_value = past_key_value[
            -2:] if past_key_value is not None else None

        coherence_rels_hidden_states = []
        coherence_rels_attention_mask = []
        device = hidden_states.device
        for rels_index, rel_embs in zip(self.coherence_rels_indices,
                                        self.coherence_rels_embs):
            if self.coherence_attn_strategy == 'SP':
                # Only use the special token embeddings
                coh_rel_embeds = rel_embs
            elif self.coherence_attn_strategy == 'Emb':
                # Only use the coherence embeddings
                coh_rel_embeds = self.coherence_embeddings(
                    torch.tensor(rels_index, dtype=torch.long, device=device))
            elif self.coherence_attn_strategy == 'SP+Emb':
                # Concatenate the coherence embeddings and special token embeddings
                coh_rel_embeds = self.coherence_embeddings(
                    torch.tensor(rels_index, dtype=torch.long, device=device))
                coh_rel_embeds = torch.cat([coh_rel_embeds, rel_embs])

            coh_rel_attn_mask = torch.ones(coh_rel_embeds.size(0),
                                           hidden_states.size(1),
                                           device=device)

            pad_size = hidden_states.size(1) - coh_rel_embeds.size(0)
            coh_rel_embeds = torch.cat([
                coh_rel_embeds,
                torch.zeros(pad_size, coh_rel_embeds.size(1), device=device)
            ],
                                       dim=0)
            coh_rel_attn_mask = torch.cat([
                coh_rel_attn_mask,
                torch.zeros(pad_size, hidden_states.size(1), device=device)
            ])

            coherence_rels_hidden_states.append(coh_rel_embeds)
            coherence_rels_attention_mask.append(coh_rel_attn_mask)

        coherence_rels_hidden_states = torch.stack(coherence_rels_hidden_states)
        coherence_rels_attention_mask = torch.stack(
            coherence_rels_attention_mask).unsqueeze(1)

        hidden_states, coherence_attn_weights, coherence_attn_present_key_value = self.coherence_attn(
            hidden_states=hidden_states,
            key_value_states=coherence_rels_hidden_states,
            attention_mask=coherence_rels_attention_mask,
            layer_head_mask=None,
            past_key_value=coherence_attn_past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.coherence_attn_layer_norm(hidden_states)

        # add coherence-attn to positions 5,6 of present_key_value tuple
        present_key_value = present_key_value + coherence_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.activation_dropout,
                                              training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value, )

        return outputs


class PersonalizedDialogueGenerator(BartForConditionalGeneration):
    config_class = PersonalizedDialogueGeneratorConfig

    PROMPT_TEMPLATE_NO_INSTRUCTION = '{response_types}'
    PROMPT_TEMPLATE = "Considering the dialogue history <query> and the user's persona traits <persona>, please generate a detailed and informative response. The response should reflect the specified response types: {response_types}, and aim to be as lengthy and comprehensive as possible to provide more details."
    CURRENT_PROMPT_TEMPLATE = PROMPT_TEMPLATE

    def __init__(self, config: PersonalizedDialogueGeneratorConfig):
        super().__init__(config)

        self.use_coherence_attention = False
        self.register_modules(config)

        self.graph_encoder_strategy = config.graph_encoder_strategy

        self.dialogue_encoder = DGatForCoherenceAwareDialogueEncoding(
            input_dim=config.dialogue_encoder_input_dim,
            hidden_dim=config.dialogue_encoder_hidden_dim,
            output_dim=config.dialogue_encoder_output_dim,
            edge_dim=config.dialogue_encoder_edge_dim,
            encoder_layer_instance=get_layer_class(
                config.dialogue_encoder_layer_type),
            num_encoder_layers=config.dialogue_encoder_num_encoder_layers,
            utterance_encoder_class=config.
            dialogue_encoder_utterance_encoder_class,
            pretrained_weights_path=config.
            dialogue_encoder_pretrained_encoder_weights_path,
            heads=config.dialogue_encoder_heads,
            add_self_loops=config.dialogue_encoder_add_self_loops,
        )

        if config.dialogue_encoder_pretrained_weights_path:
            self.dialogue_encoder.load_state_dict(torch.load(
                config.dialogue_encoder_pretrained_weights_path),
                                                  strict=False)

        if config.freeze_dialogue_encoder:
            for param in self.dialogue_encoder.parameters():
                param.requires_grad = False

            self.dialogue_encoder.eval()

        self.projector = nn.Linear(config.dialogue_encoder_output_dim,
                                   config.d_model,
                                   bias=False)

        self.weights_decider = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model), nn.Sigmoid())

        # Multiple Choice Head
        # Binary Classification: True is the correct answer, False is the incorrect answer
        self.classification_head = SequenceSummary(config)

        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens(NEW_SPEICAL_TOKENS)

        self.max_length = MAX_MODEL_SEQ_LENGTH

    def add_soft_prompt(self, n_tokens=20):
        self.n_tokens = n_tokens
        embedder = SoftPromptEmbedder(wte=self.model.shared,
                                      n_tokens=self.n_tokens,
                                      initialize_from_vocab=True)
        self.model.set_input_embeddings(embedder)

        self.mix_soft_and_hard_prompt = False

    def register_modules(self,
                         config: PersonalizedDialogueGeneratorConfig) -> None:
        # Change the decoder layers to `CoherenceAwareBartDecoderLayer`
        self.model.decoder.layers = nn.ModuleList([
            CoherenceAwareBartDecoderLayer(config)
            for _ in range(config.decoder_layers)
        ])
        self.use_coherence_attention = True

    def forward(self,
                dialogue_encoder_input: dict[torch.Tensor],
                generator_input: dict[torch.Tensor],
                top_k_relations: Optional[int] = 3,
                tau: Optional[float] = 0.2,
                use_resp_type_prediction_in_training: Optional[bool] = False,
                is_generation: Optional[bool] = False,
                **kwargs) -> PersonalizedDialogueGeneratorOutput:
        device = generator_input['input_ids'].device

        dial_encoder_outputs = self.dialogue_encoder(
            **dialogue_encoder_input,
            use_resp_type_prediction_in_training=
            use_resp_type_prediction_in_training)

        coherence_relation_preds = dial_encoder_outputs.last_resp_type_seq_preds \
            if not self.training or use_resp_type_prediction_in_training \
            else dial_encoder_outputs.last_resp_type_labels

        coherence_relation_preds_label, coherence_relation_preds_index, \
            coherence_relation_preds_embs = [], [], []
        for batch_idx, probs in enumerate(coherence_relation_preds):
            # If not training, we consider only the top-k predictions
            # Otherwise, we use all the ground-truth labels
            if not self.training:
                top_k_labels = [
                    COHERENCE_RELATIONS[index]
                    for index in torch.topk(probs, top_k_relations)[1]
                ]
            else:
                top_k_labels = [
                    COHERENCE_RELATIONS[label]
                    for label in probs.nonzero(as_tuple=False).squeeze(1)
                ]

                if len(top_k_labels) == 0:
                    # If ground-truth labels are empty (all zeros),
                    # we use the top-k predictions
                    probs = dial_encoder_outputs.next_resp_type_seq_preds[
                        batch_idx]
                    top_k_labels = [
                        COHERENCE_RELATIONS[index]
                        for index in torch.topk(probs, top_k_relations)[1]
                    ]

            coherence_relation_preds_label.append(top_k_labels)
            coherence_relation_preds_index.append(
                [COHERENCE_RELATIONS.index(label) for label in top_k_labels])

            rel_tokens = self.tokenizer(' '.join(
                [f'<{label.lower()}> {label}' for label in top_k_labels]),
                                        add_special_tokens=False,
                                        return_tensors='pt').input_ids[0]
            coherence_relation_preds_embs.append(
                self.model.decoder.embed_tokens(rel_tokens.to(device)))

        # Discrete Prompt
        prompt_inputs = [
            self.format_prompt(rels, self.CURRENT_PROMPT_TEMPLATE)
            for rels in coherence_relation_preds_label
        ]
        prompt_encodings = self.tokenizer(prompt_inputs,
                                          add_special_tokens=False,
                                          padding='max_length',
                                          truncation=True,
                                          return_tensors='pt')

        lm_labels = []
        decoder_input_ids = []
        decoder_attention_mask = []
        for prompt_input_ids, response_input_ids in zip(
                self.remove_padding_tokens(prompt_encodings.input_ids,
                                           prompt_encodings.attention_mask),
                self.remove_padding_tokens(
                    generator_input['decoder_input_ids'],
                    generator_input['decoder_attention_mask'])):
            num_prompt_tokens = len(prompt_input_ids)
            num_response_tokens = len(response_input_ids)
            pad_size = self.max_length - num_prompt_tokens - num_response_tokens

            if not is_generation:
                decoder_input_ids.append(
                    torch.cat([
                        prompt_input_ids.to(device),
                        response_input_ids.to(device),
                        torch.tensor([self.tokenizer.pad_token_id] * pad_size,
                                     device=device)
                    ]))
                decoder_attention_mask.append(
                    torch.cat([
                        torch.ones(num_prompt_tokens),
                        torch.ones(num_response_tokens),
                        torch.zeros(pad_size)
                    ]))
                lm_labels.append(
                    torch.cat([
                        torch.tensor([-100] * num_prompt_tokens, device=device),
                        response_input_ids[1:].to(device),
                        torch.tensor([-100] * (pad_size + 1), device=device)
                    ]))
            else:
                decoder_input_ids.append(
                    torch.cat([
                        prompt_input_ids.to(device),
                        response_input_ids.to(device)
                    ]))
                decoder_attention_mask.append(
                    torch.cat([
                        torch.ones(num_prompt_tokens),
                        torch.ones(num_response_tokens)
                    ]))

        decoder_input_ids = torch.stack(decoder_input_ids)
        decoder_attention_mask = torch.stack(decoder_attention_mask)
        generator_input['decoder_input_ids'] = decoder_input_ids.to(device)
        generator_input['decoder_attention_mask'] = decoder_attention_mask.to(
            device)

        if not is_generation:
            lm_labels = torch.stack(lm_labels)
            cls_labels = generator_input['multiple_choice_label']
            del generator_input['multiple_choice_label']
        else:
            lm_labels = None
            cls_labels = None

            if 'multiple_choice_label' in generator_input:
                del generator_input['multiple_choice_label']

        if self.use_coherence_attention:
            for layer in self.model.decoder.layers:
                layer.coherence_rels_indices = coherence_relation_preds_index
                layer.coherence_rels_embs = coherence_relation_preds_embs

        text_encoder_decoder_outputs = self.model(**generator_input,
                                                  encoder_outputs=None,
                                                  output_attentions=True)

        if self.graph_encoder_strategy != 'None':
            if self.graph_encoder_strategy == 'Attn':
                dial_encoder_hidden_states = dial_encoder_outputs.personalized_graph_embeddings
            elif self.graph_encoder_strategy == 'Add':
                dial_encoder_hidden_states = dial_encoder_outputs.context_graph_embeddings + \
                    dial_encoder_outputs.persona_graph_embeddings
            elif self.graph_encoder_strategy == 'C':
                dial_encoder_hidden_states = dial_encoder_outputs.context_graph_embeddings
            elif self.graph_encoder_strategy == 'P':
                dial_encoder_hidden_states = dial_encoder_outputs.persona_graph_embeddings
            elif self.graph_encoder_strategy == 'Random':
                dial_encoder_hidden_states = torch.rand_like(
                    dial_encoder_outputs.personalized_graph_embeddings,
                    device=device)

            dial_encoder_decoder_outputs = self.model(
                attention_mask=torch.ones_like(
                    generator_input['decoder_attention_mask']),
                decoder_input_ids=generator_input['decoder_input_ids'],
                decoder_attention_mask=generator_input[
                    'decoder_attention_mask'],
                encoder_outputs=(self.projector(
                    dial_encoder_hidden_states).unsqueeze(1).expand(
                        -1, self.max_length if not is_generation else
                        generator_input['decoder_attention_mask'].size(1),
                        -1), None, None),
                output_attentions=True)

            text_encoder_decoder_hidden_states = text_encoder_decoder_outputs.last_hidden_state
            dial_encoder_decoder_hidden_states = dial_encoder_decoder_outputs.last_hidden_state
            residual = text_encoder_decoder_hidden_states + dial_encoder_decoder_hidden_states
            # Dynamic Weighted Aggregation
            text_encoder_alpha = self.weights_decider(
                torch.cat([
                    text_encoder_decoder_hidden_states,
                    dial_encoder_decoder_hidden_states
                ],
                          dim=-1))

            text_encoder_alpha_mask = []
            dial_encoder_alpha_mask = []
            for alpha in text_encoder_alpha:
                text_encoder_mask = torch.where(alpha > tau,
                                                torch.ones_like(alpha),
                                                torch.zeros_like(alpha))
                text_encoder_alpha_mask.append(text_encoder_mask)

                dial_encoder_mask = torch.where(alpha < 1 - tau,
                                                torch.ones_like(alpha),
                                                torch.zeros_like(alpha))
                dial_encoder_alpha_mask.append(dial_encoder_mask)

            text_encoder_alpha_mask = torch.stack(text_encoder_alpha_mask)
            dial_encoder_alpha_mask = torch.stack(dial_encoder_alpha_mask)
            weighted_hidden_states = text_encoder_alpha * \
                    text_encoder_alpha_mask * text_encoder_decoder_hidden_states \
                    + (1 - text_encoder_alpha) * \
                    dial_encoder_alpha_mask * dial_encoder_decoder_hidden_states

            outputs = residual + weighted_hidden_states
        else:
            outputs = text_encoder_decoder_outputs.last_hidden_state

        lm_logits = self.lm_head(outputs)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        # Language Modeling
        masked_lm_loss = None
        if lm_labels is not None:
            lm_labels = lm_labels.to(lm_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))

        # Multiple Choice
        cls_logits = None
        cls_loss = None
        if cls_labels is not None:
            eos_mask = generator_input['decoder_input_ids'].eq(
                self.config.eos_token_id).to(outputs.device)
            eos_indices = torch.argmax(eos_mask.to(torch.int), dim=1)

            cls_logits = self.classification_head(outputs,
                                                  eos_indices).squeeze(-1)

            loss_fct = nn.CrossEntropyLoss()
            cls_labels = cls_labels.to(cls_logits.device).to(torch.long)
            cls_loss = loss_fct(cls_logits.view(-1, cls_logits.size(-1)),
                                cls_labels.view(-1))

        if is_generation:
            num_beams = self.config.task_specific_params['generation']['num_beams'] \
                if 'generation' in self.config.task_specific_params else 10
            lm_logits = lm_logits.expand(num_beams, -1, -1)

        return PersonalizedDialogueGeneratorOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            cls_loss=cls_loss,
            cls_logits=cls_logits,
            cls_labels=cls_labels.to(torch.float)
            if not is_generation else None,
            dial_encoder_outputs=dial_encoder_outputs)

    def remove_padding_tokens(self, input_ids: torch.Tensor,
                              masks: torch.Tensor) -> torch.Tensor:
        return [tokens[mask == 1] for tokens, mask in zip(input_ids, masks)]

    def format_prompt(self, rels: list[str], prompt: str) -> str:
        if len(rels) >= 3:
            response_types = ''
            for rel in rels[:-1]:
                response_types += f'{NEW_SPEICAL_TOKENS_MAP[rel.lower()]} {rel}, '
            response_types += f'and {NEW_SPEICAL_TOKENS_MAP[rels[-1].lower()]} {rels[-1]}'
        elif len(rels) == 2:
            response_types = f'{NEW_SPEICAL_TOKENS_MAP[rels[0].lower()]} {rels[0]}'
            response_types += f' and {NEW_SPEICAL_TOKENS_MAP[rels[1].lower()]} {rels[1]}'
        else:
            response_types = f'{NEW_SPEICAL_TOKENS_MAP[rels[0].lower()]} {rels[0]}'

        return prompt.format(response_types=response_types)

    def compute_loss(
        self,
        outputs: PersonalizedDialogueGeneratorOutput,
        weight: dict[dict, float] = None
    ) -> PersonalizedDialogueGeneratorTrainerOutput:
        total_loss = outputs.loss + outputs.cls_loss

        losses = PersonalizedDialogueGeneratorTrainerOutput(
            total_loss=total_loss,
            nll_loss=outputs.loss,
            cls_loss=outputs.cls_loss)

        if not self.config.freeze_dialogue_encoder:
            dial_encoder_outputs = self.dialogue_encoder.compute_loss(
                outputs.dial_encoder_outputs, weight=weight)

            total_loss += dial_encoder_outputs.total_loss.item()
            losses.total_loss = total_loss

            losses.dial_encoder_coh_rel_cls_loss = dial_encoder_outputs.coh_rel_cls_loss
            losses.dial_encoder_next_resp_type_direct_loss = dial_encoder_outputs.next_resp_type_direct_loss
            losses.dial_encoder_next_resp_type_seq_loss = dial_encoder_outputs.next_resp_type_seq_loss
            losses.dial_encoder_link_prediction_loss = dial_encoder_outputs.link_prediction_loss

        return losses

    def prepare_inputs_for_generation(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            **kwargs):
        dialogue_encoder_input = kwargs.get('dialogue_encoder_input', None)
        if dialogue_encoder_input is None:
            raise ValueError('dialogue_encoder_input must be provided')

        generator_input = kwargs.get('generator_input', {})
        if 'input_ids' not in generator_input:
            raise ValueError(
                'generator_input must contain input_ids for Encoder')
        if input_ids is not None:
            generator_input['decoder_input_ids'] = input_ids
            generator_input[
                'decoder_attention_mask'] = decoder_attention_mask.expand(
                    -1, input_ids.size(1))

        return {
            'dialogue_encoder_input': dialogue_encoder_input,
            'generator_input': generator_input,
            'top_k_relations': kwargs.get('top_k_relations', 3),
            'tau': kwargs.get('tau', 0.2),
            'use_resp_type_prediction_in_training': True,
            'is_generation': True
        }
