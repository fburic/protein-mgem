# Various monkeypatches to fix issues with TAPE code
from packaging import version

import torch


if version.parse(torch.__version__) > version.parse('1.4'):
    from tape.registry import registry
    from tape.models.modeling_bert import (ProteinBertModel,
                                           ProteinBertAbstractModel,
                                           ValuePredictionHead)


    @registry.register_task_model('embed', 'transformer_parallel')
    class ProteinBertModel_parallel(ProteinBertModel):
        def forward(self,
                    input_ids,
                    input_mask=None):
            """
            nn.Model.parameters() on models replicated across different GPUs
            are no longer populated since PyTorch ?1.5.0,
            raising a StopIteration exception.

            https://github.com/vid-koci/bert-commonsense/issues/6
            https://github.com/huggingface/transformers/issues/3936
            https://github.com/pytorch/pytorch/issues/40457
            """
            if input_mask is None:
                input_mask = torch.ones_like(input_ids)

            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)

            # Since input_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.

            # FIX =====
            # <<<<<
            # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            # >>>>>
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
            # =========

            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            embedding_output = self.embeddings(input_ids)
            encoder_outputs = self.encoder(embedding_output,
                                           extended_attention_mask,
                                           chunks=None)
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output)

            # add hidden_states and attentions if they are here
            outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
            return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


    @registry.register_task_model('fluorescence', 'transformer_parallel')
    @registry.register_task_model('stability', 'transformer_parallel')
    class ProteinBertForValuePrediction_parallel(ProteinBertAbstractModel):

        def __init__(self, config):
            super().__init__(config)

            self.bert = ProteinBertModel_parallel(config)
            self.predict = ValuePredictionHead(config.hidden_size)

            self.init_weights()

        def forward(self, input_ids, input_mask=None, targets=None):
            outputs = self.bert(input_ids, input_mask=input_mask)

            sequence_output, pooled_output = outputs[:2]
            outputs = self.predict(pooled_output, targets) + outputs[2:]
            # (loss), prediction_scores, (hidden_states), (attentions)
            return outputs
