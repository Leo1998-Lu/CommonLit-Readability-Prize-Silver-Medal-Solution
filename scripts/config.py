class Config:
    model_name = 'roberta-large'
    output_hidden_states = True
    epochs = 3
#     evaluate_interval = 40
    batch_size = 8
    device = 'cuda'
    seed = 2021
    max_len = 248
    lr = 1e-5
    wd = 0.01
#     eval_schedule = [(float('inf'), 40), (0.5, 30), (0.49, 20), (0.48, 10), (0.47, 3), (0, 0)]
    eval_schedule = [(float('inf'), 40), (0.47, 20), (0.46, 10), (0, 0)]

    gradient_accumulation = 2