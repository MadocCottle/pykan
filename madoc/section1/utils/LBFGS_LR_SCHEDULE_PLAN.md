# Learning Rate Scheduling for LBFGS Optimizer - Implementation Plan

## Background

LBFGS is a second-order optimization method that doesn't use a traditional learning rate in the same way as first-order optimizers (Adam, SGD). However, PyTorch's LBFGS implementation does support a learning rate parameter (default: 1.0) that scales the step size.

## Current Issue

LBFGS can struggle with:
1. **Overshooting** - Taking steps that are too large, especially early in training
2. **Plateau detection** - Continuing to take full steps even when loss has plateaued
3. **Numerical instability** - Large steps can cause gradient explosions

## Proposed Solutions

### Option 1: Adaptive Step Size (Recommended)

Modify the LBFGS optimizer to use a decaying learning rate based on progress:

```python
def train_model_with_adaptive_lr(model, dataset, epochs, device,
                                  initial_lr=1.0, lr_decay=0.9,
                                  patience=3, min_lr=0.001):
    """
    Train with adaptive learning rate for LBFGS

    Args:
        initial_lr: Starting learning rate for LBFGS (default: 1.0)
        lr_decay: Multiplicative decay factor (default: 0.9)
        patience: Epochs to wait before reducing lr (default: 3)
        min_lr: Minimum learning rate (default: 0.001)
    """
    optimizer = torch.optim.LBFGS(model.parameters(), lr=initial_lr, max_iter=20)

    best_loss = float('inf')
    epochs_without_improvement = 0
    current_lr = initial_lr

    for epoch in range(epochs):
        # Check if we should reduce learning rate
        if epochs_without_improvement >= patience and current_lr > min_lr:
            current_lr = max(current_lr * lr_decay, min_lr)
            # Recreate optimizer with new lr
            optimizer = torch.optim.LBFGS(model.parameters(), lr=current_lr, max_iter=20)
            epochs_without_improvement = 0
            print(f"  Reducing LR to {current_lr:.6f}")

        # Training step...
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(dataset['train_input']), dataset['train_label'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            return loss

        optimizer.step(closure)

        # Track improvement
        current_loss = train_losses[-1]
        if current_loss < best_loss * 0.999:  # 0.1% improvement threshold
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
```

### Option 2: Scheduled Learning Rate Decay

Use a predefined schedule:

```python
def get_lbfgs_lr_schedule(epoch, initial_lr=1.0):
    """
    Returns learning rate for given epoch

    Schedules:
    - Epochs 0-2: initial_lr (warm-up with full steps)
    - Epochs 3-5: initial_lr * 0.5 (moderate steps)
    - Epochs 6+: initial_lr * 0.1 (fine-tuning)
    """
    if epoch < 3:
        return initial_lr
    elif epoch < 6:
        return initial_lr * 0.5
    else:
        return initial_lr * 0.1
```

### Option 3: Strong Wolfe Line Search (Advanced)

LBFGS already supports strong Wolfe line search internally via the `line_search_fn` parameter:

```python
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=20,
    line_search_fn='strong_wolfe'  # Enable adaptive step sizing
)
```

This makes the optimizer automatically choose appropriate step sizes.

## Recommended Implementation Strategy

**For Section 1 experiments:**

1. **Primary approach**: Use Option 3 (strong_wolfe) as it's built-in and well-tested
2. **Fallback**: If strong_wolfe causes issues, implement Option 1 (adaptive lr)
3. **For comparison**: Keep current default behavior as baseline

## Implementation Steps

1. Add `line_search_fn='strong_wolfe'` to LBFGS optimizer in `train_model()`
2. Make it optional via a parameter (default: None for backward compatibility)
3. Add a new function `train_model_adaptive_lr()` for Option 1
4. Document the differences in training behavior

## Expected Benefits

- **Reduced gradient explosions**: Smaller steps when gradients are large
- **Better convergence**: Adaptive steps help avoid overshooting
- **Faster training**: Can use larger steps early, smaller steps for fine-tuning
- **More stable training**: Automatic detection of when to slow down

## Compatibility Considerations

- Must maintain backward compatibility with existing experiments
- Should be opt-in (off by default) until validated
- Need to test on all activation types (tanh, relu, silu)
- Document performance differences

## Testing Plan

1. Run small test on single dataset with all three options
2. Compare convergence speed and final loss
3. Check for any numerical instabilities
4. If stable, integrate into main training pipeline

## References

- PyTorch LBFGS docs: https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
- Strong Wolfe conditions: Numerical optimization theory for line search
- Nocedal & Wright, "Numerical Optimization" (2006) - Chapter 7
