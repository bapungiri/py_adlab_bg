from banditpy.models.rnn_models import TinyBehaviorRNN, TinyBehaviorRNNTrainer
import numpy as np


# Build some synthetic sessions
def gen_session(T=200, p=(0.3, 0.7), seed=None):
    rng = np.random.default_rng(seed)
    actions = []
    rewards = []
    last = rng.integers(0, 2)
    for t in range(T):
        # simple biased policy
        if rng.random() < 0.1:  # exploration
            a = rng.integers(0, 2)
        else:
            a = last
        r = 1 if rng.random() < p[a] else 0
        actions.append(a)
        rewards.append(r)
        last = a if r == 1 else last
    return {"actions": np.array(actions), "rewards": np.array(rewards)}


train_sessions = [gen_session(seed=i) for i in range(20)]
val_sessions = [gen_session(seed=100 + i) for i in range(5)]
test_sessions = [gen_session(seed=200 + i) for i in range(5)]

model = TinyBehaviorRNN(
    input_size=3, num_actions=2, hidden_size=2, diagonal_readout=False
)
trainer = TinyBehaviorRNNTrainer(
    model, lr=5e-3, weight_decay=5e-4, patience=15, max_epochs=300
)

history = trainer.fit(train_sessions, val_sessions)
metrics = trainer.evaluate(test_sessions)
print("Final test NLL:", metrics["test_nll"])
print("Train NLL history (last 5):", history["train_nll"][-5:])
print("Val NLL history (last 5):", history["val_nll"][-5:])
