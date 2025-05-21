# MM21B029 - Assignment 3

##  Environment Setup

1. **Clone this repository** and navigate into the folder.
2. **Create a virtual environment** and install dependencies:
    ```bash
    python -m venv env
    source env/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```
3. **Download the Dakshina Dataset v1.0** and place it in the root directory or update the path in `load_data.py`.

---

##  Attention Model: Usage

Edit the model configuration as required from attention_model.py

```bash

python attention_model.py

```

##  Vanilla Model: Usage

Edit the model configuration as required from vanilla_model.py

```bash

python vanilla_model.py

```

---

##  WandB Logging

Both models support hyperparameter sweeps via `wandb`.

```bash
# Edit sweep_config inside vanilla_sweep.py or attention_sweep.py
python attention_sweep.py
python vanilla_sweep.py
```

