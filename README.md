# Train_GPT

## Modify `model.py` and `train.py`
- In `model.py`： Complete the __init__() method for model construction using torch.nn.
- In `train.py`： Modify the model initialization to use three different models: `transformer`, `RNN`, and `LSTM`.

## Result0

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; justify-items: center; align-items: start;">
    <figure style="margin: 0; padding: 0;">
        <img src="./pic/arg0.png" alt="First Train args" style="display: block; max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; padding: 4px; box-sizing: border-box;">
        <figcaption style="text-align: center; font-style: italic; font-size: 1.5em; color: #001; margin-top: 10px; text-shadow: 1px 1px 5px rgba(255, 255, 255, 0.8);">First Train args</figcaption>
    </figure>
    <figure style="margin: 0; padding: 0;">
        <img src="./pic/transformer_arg0.png" alt="Transformer Train args" style="display: block; max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; padding: 4px; box-sizing: border-box;">
        <figcaption style="text-align: center; font-style: italic; font-size: 1.5em; color: #001; margin-top: 10px; text-shadow: 1px 1px 5px rgba(255, 255, 255, 0.8);">Transformer Train args</figcaption>
    </figure>
    <figure style="margin: 0; padding: 0;">
        <img src="./pic/RNN_arg0.png" alt="RNN Train args" style="display: block; max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; padding: 4px; box-sizing: border-box;">
        <figcaption style="text-align: center; font-style: italic; font-size: 1.5em; color: #001; margin-top: 10px; text-shadow: 1px 1px 5px rgba(255, 255, 255, 0.8);">RNN Train args</figcaption>
    </figure>
    <figure style="margin: 0; padding: 0;">
        <img src="./pic/LSTM_arg0.png" alt="LSTM Train args" style="display: block; max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; padding: 4px; box-sizing: border-box;">
        <figcaption style="text-align: center; font-style: italic; font-size: 1.5em; color: #001; margin-top: 10px; text-shadow: 1px 1px 5px rgba(255, 255, 255, 0.8);">LSTM Train args</figcaption>
    </figure>
</div>

## TODO
- [ ] Choose one of the three models: `transformer`, `RNN`, or `LSTM` and implement it from scratch.