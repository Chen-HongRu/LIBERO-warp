from libero.lifelong._dependencies import require_legacy_dependencies

require_legacy_dependencies()

from libero.lifelong.models.bc_rnn_policy import BCRNNPolicy
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
from libero.lifelong.models.bc_vilt_policy import BCViLTPolicy

from libero.lifelong.models.base_policy import get_policy_class, get_policy_list
