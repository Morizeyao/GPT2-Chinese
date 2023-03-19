from abc import ABC

from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import (
    BaseModelOutputWithPastAndCrossAttentions,
    torch,
    Optional, Tuple, Union
)


class GPT2ModelWithAdditionalHiddenStates(GPT2Model, ABC):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, additional_hidden_states: Optional[torch.FloatTensor] = None,
                **kwargs) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        :param additional_hidden_states: shape = [batch, seq_length, embed_size]
        """
        if 'input_ids' in kwargs:
            input_ids = kwargs.pop('input_ids')
            inputs_embeds: torch.FloatTensor = self.wte(input_ids)
        else:
            inputs_embeds: torch.FloatTensor = kwargs.pop('inputs_embeds')

        assert inputs_embeds.size() == additional_hidden_states.size()

        kwargs['inputs_embeds'] = inputs_embeds + additional_hidden_states

        return super().forward(**kwargs)


class GPT2ModelWithContext(GPT2ModelWithAdditionalHiddenStates, ABC):
    def __init__(self, config):
        super().__init__(config)

    def context_to_embedding(self, context: dict) -> torch.FloatTensor:
        """
        :return: additional_hidden_states, shape = [batch, seq_length, embed_size]
        """
        raise NotImplementedError

    def forward(self, context: dict = None, **kwargs) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        additional_hidden_states: torch.FloatTensor = self.context_to_embedding(context)
        return super().forward(additional_hidden_states, **kwargs)
