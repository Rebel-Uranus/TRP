# Truncated Return Prediction
[dmcontrol-generalization-benchmark](https://github.com/nicklashansen/dmcontrol-generalization-benchmark)

This project conducts research in [TRP](https://ojs.aaai.org/index.php/AAAI/article/view/28369). Our experimental design, baselines, and environment dependencies are all inherited from the work of [dmcontrol-generalization-benchmark](https://github.com/nicklashansen/dmcontrol-generalization-benchmark). We adopt the same datasets, evaluation metrics, and baseline methods as theirs to ensure result comparability. The software and hardware requirements also follow their specifications. We sincerely thank them for their significant contributions to this area, and recommend referring to their work for detailed information.

## Training & Evaluation

The `scripts` directory contains training and evaluation bash scripts for all the included algorithms. Alternatively, you can call the python scripts directly, e.g. for training call

```
python3 src/train.py \
  --algorithm trp \
  --aux_lr 3e-4 \
  --seed 0
```

to run TRP on the default task, `walker_run`. This should give you an output of the form:

```
Working directory: logs/walker_walk/svea/0
Evaluating: logs/walker_walk/svea/0
| eval | S: 0 | ER: 26.2285 | ERTEST: 25.3730
| train | E: 1 | S: 250 | D: 70.1 s | R: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | AUXLOSS: 0.0000
```
where `ER` and `ERTEST` corresponds to the average return in the training and test environments, respectively. You can select the test environment used in evaluation with the `--eval_mode` argument, which accepts one of `(train, color_easy, color_hard, video_easy, video_hard, distracting_cs, none)`. Use `none` if you want to disable continual evaluation of generalization. Note that not all combinations of arguments have been tested. Feel free to open an issue or send a pull request if you encounter an issue or would like to add support for new features.

