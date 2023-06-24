### tf.keras.losses.BinaryCrossentropy
- from_logits=True 
    - the predicted values are not bounded between 0 and 1, and they are considered as unnormalized logits or raw model outputs
    - the sigmoid activation is applied internally by the loss function.
- from_logits=False
    - the predicted values are assumed to be already probabilities, and no additional activation is applied.

### Saving models
Model weights and model configuration are different.
2 Saving Options:

1. Save weight + config(metrics..) => package into the full model
2. Save weight  (if we are able to get config)