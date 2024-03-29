# Category-Based Strategy-Driven Question Generator for Visual Dialogue

## 摘要
GuessWhat?! is a task-oriented visual dialogue task which has two players, a guesser and an oracle.
Guesser aims to locate the object supposed by oracle by asking several Yes/No questions which are answered by oracle.
How to ask proper questions is crucial to achieve the final goal of the whole task.
Previous methods generally use an end-to-end generator with an implicit question generating strategy, which is usually hard for the models to grasp the efficient questioning strategy without a proper structure.
This makes generated questions too simple and easily useless for the final purpose.
This paper proposes a category-based strategy-driven question generator(CSQG) to explicitly provide a category based questioning strategy for the generator.
First we encode the image and the dialogue history and decide the category of question to focus at the current step. Then the question is generated by adding the context and this proper category.
The evaluation on large-scale visual dialogue dataset GueesWhat?! shows that our method can help guesser achieve 51.71\% success rate which is the state-of-the-art on the supervised training methods.

## 训练
### QGen
```
python train_qgen.py
```

### Oracle
```
python train_oracle.py
```

### Guesser
```
python train_guesser.py
```