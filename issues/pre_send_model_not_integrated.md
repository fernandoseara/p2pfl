# PreSendModelCommand Not Integrated

## Status

**DISABLED** - File converted to docstring to fix circular imports.

## Location

`p2pfl/communication/commands/message/pre_send_model_command.py`

## What Happened

The file was causing a circular import chain:

```
gossiper.py → PreSendModelCommand → FullModelCommand → BasicLearningWorkflowModel → ... → circular
```

To break this chain, the file's code was converted into a docstring, effectively disabling it while preserving the implementation for future reference.

## Purpose (Intended)

Bandwidth optimization command to notify recipient nodes before sending model weights:

1. Sender broadcasts `PRE_SEND_MODEL` with model hash
2. Recipient checks if it already has this model
3. Recipient responds `"true"` (send it) or `"false"` (already have it)
4. Sender only sends weights if recipient needs them

## Why It Was Never Integrated

- Not registered in any workflow builder
- Not called from any stage or workflow
- Uses state fields that may not exist (`sending_models`, `train_set`, `models_aggregated`)

## Current State

The entire implementation is wrapped in a triple-quoted docstring. The file contains:
- Original deprecation warnings
- Complete implementation logic (commented out)
- Spanish comment: "NO ESTA METIDO" (not included)

## Decision Needed

1. **Delete the file** - If bandwidth optimization is not needed
2. **Re-enable and integrate** - If the feature is desired:
   - Uncomment the code
   - Fix imports to avoid circular dependencies
   - Add to workflow builders
   - Add required state fields
   - Integrate into gossip flow

## Impact on gossiper.py

`gossiper.py` still imports `PreSendModelCommand` at line 26. This import now effectively does nothing since the class is commented out. The import should be removed or the gossiping logic that uses `PreSendModelCommand.get_name()` should be updated.

## Affected Files

- `p2pfl/communication/commands/message/pre_send_model_command.py` - Disabled
- `p2pfl/communication/protocols/protobuff/gossiper.py` - Has unused import
