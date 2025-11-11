# Vision-Based Driving Agent (SuperTuxKart)

A CNN-driven autonomous driving agent that maps raw frames → steering, acceleration, braking using a **Planner** (perception + aim-point head) and a **Controller** (low-level actions). Built with PyTorch and integrated with SuperTuxKart via PySuperTuxKart.

> Evaluated on **6 unique courses** with performance **on par with skilled human players**.

---

## ✨ Highlights

- **Planner**: UNet-style encoder–decoder with residual blocks and a **soft-argmax** head predicting an image-space aim point (`src/planner.py`).
- **Controller**: Rule-based controller using aim-point + current velocity to set steer/throttle/brake/drift (`src/controller.py`).
- **Simulator Integration**: Rollouts, dataset collection, and camera-space projection helpers (`src/utils.py`).
- **Reproducible Training**: Minimal, readable training loop with TensorBoard hooks (`src/train.py`).

---

## 🛠️ Setup

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate    

pip install -r requirements.txt