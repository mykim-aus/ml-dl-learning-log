<sub>🌐 **English** · [한국어](../ko/study-notes.md)</sub>

# Learning Machine Learning by Asking an AI, Line by Line

> A growing log of everything I ask AI while learning machine learning and deep learning — **studying since June 29, 2026**. New topics are appended to this same file over time.

## Why I tried this

I saw this post by Andrej Karpathy:

![Andrej Karpathy — how to become expert at a thing](../assets/karpathy-how-to-become-expert.webp)

> How to become expert at thing:
>
> 1. iteratively take on concrete projects and accomplish them depth wise, learning "on demand" (ie don't learn bottom up breadth wise)
> 2. teach/summarize everything you learn in your own words
> 3. only compare yourself to younger you, never to others

So I decided to try it.

---

## Deep Learning & GPU

> Machine learning basics → GPU hardware → how memory works → the actual training process → the principles behind weights and derivatives → LLMs and recent trends: these are notes that pull a single session's worth of conversation into one continuous thread.

---

## The One-Line Core (a conclusion I worked out myself)

> You have measurements and a corresponding answer key, and you start the weights off at random.
> Multiply measurement × weight to **make a prediction (forward pass)**, then **measure how far that prediction is from the answer (loss).**
> **Use differentiation to find the gradient (direction + magnitude)**, then **nudge the weights a little in that direction, by an amount set by the learning rate.**
> Run it again with the new weights → **repeat until you land near the weights with the smallest loss.**
>
> And this loop is **identical everywhere — images, video, text, LLMs.**
> The only things that change from field to field are: ① how you turn the input into numbers, ② how you define the answer (loss), and ③ the model architecture.

---

> ⚠️ **This was about linear regression (the deep-learning family that learns via gradient descent).**
> The loop above fits linear regression, neural networks, CNNs, and LLMs — but it's **not how all machine learning works.**
> Decision trees, KNN, and others don't use weights or differentiation; they learn in other ways.

> **2026-07-01 — I realized that depending on the model (the "frame"), there may be no weights and no layers at all.**
> Layers, weight sets, and ReLU are all **components exclusive to the neural-network family.** A tree learns through questions (branches).

---

## The Questions Covered in This Session

**Machine Learning / Deep Learning Basics**

1. The concepts of machine learning and deep learning

2. Why GPUs and memory matter in deep learning

3. Forward pass and backpropagation (the easy explanation)

4. Whether a Research Engineer needs to understand semiconductor design

**GPU Hardware and Memory**

5. How the GPU and memory actually run deep learning computations

6. Whether you can load straight onto the GPU without the CPU

7. Whether VRAM, cache, registers, and cores are separate physical parts

8. Whether the order CPU preprocessing → VRAM → L2 → L1 → register → core is correct

9. How a core decides it "needs a value" (the source code)

**How Data and Models Get Onto the GPU**

10. How processing works when the data is 100GB but VRAM is 24GB

11. Whether the model also goes onto the GPU, and whether "model" means the trained model

12. Whether the GPU is used for both training and inference

13. The full name of VRAM and how it differs from regular RAM

**The Actual Principles of Training**

14. Whether weight training is always matrix multiplication

15. How the image and the answer get moved to the GPU

16. Whether preprocessing is always width × height × channels, and whether the model judges by shape rather than color

17. Whether you use an existing CNN, and who sets the pixel-color-difference criterion

18. How you tell the model that this is "classification training" (labels)

19. Whether there are models without labels

**The Principles of Weights, Gradients, and Differentiation**

20. What a gradient is

21. Whether you can think of a weight as the "probability of being the right answer"

22. Whether the model decides the weights on its own

23. The principle of finding the weights via matrix multiplication

24. The basis on which backpropagation issues its correction instructions

25. The principle of how differentiation knows the effect on the loss (without math)

26. How you confirm things got better using the gradient from differentiation

27. The basis of differentiation's judgment, and how the input and weight values are set

28. Whether deep learning is ultimately the process of finding good weights

**The Repetition and Ending of Training**

29. Whether the structure is that the cores keep running matrix multiplication

30. Whether more iterations always means more accuracy

31. The basis for deciding "a lot vs. a moderate amount"

**Types of Learning (Going Deeper)**

32. What the inputs are for unsupervised and self-supervised learning

**Throwing Numbers at It vs. Differentiation, and Math**

33. Since differentiation is also just computing numbers, couldn't you just throw any number at it and test?

34. When a new weight comes out, do you search over combinations including all the old weights too?

35. Couldn't you just throw all the values around the differentiation answer (0.27, 0.29, 0.31) at it?

36. Is it ultimately a matter of time and cost?

37. Is math like calculus and linear algebra essential for deep learning? (※ read alongside #31)

**The Principles and Limits of a Model's Judgment**

38. Does the model infer features like "ear shape" on its own from the answers?

39. Can you know what a finished model bases its judgments on?

**Computing It by Hand (Flower Classification Practice)**

40. Does one measurement always have exactly one weight? / If there are 3 measurements and 3 outputs, are there 9 weights?

41. Where do the numbers for the answer and prediction come from? / What values go into matrices A and B?

42. Why does the loss square the difference? (Is it a convention, or for efficiency?)

43. Isn't a larger measurement more influential? → normalization

44. Who sets the normalized value (2→0.2), and on what basis? / Is there an automatic library?

45. How are all 9 weights differentiated at once, and what comes out as the result of differentiation?

46. Is there a fixed ratio for the "more/less" adjustment based on the gradient? (learning rate)

47. Is differentiation about finding the gradient? / What is the differentiation formula?

48. How do the measurements and answers go into the differentiation formula?

49. Is the gradient the same thing as the weight?

50. Can I understand it as a graph of the loss approaching the answer? (with PyTorch verification)

**LLMs and Recent Trends**

51. LLMs have no answer key, so how do they compute the loss?

52. Do you turn text into numbers (tokenization, embeddings)? / Is the dictionary prepared in advance? / The principle behind how embeddings cluster / Are embeddings also models?

53. Can you just learn the model architecture field by field? / How do you read papers?

54. Do the basic deep learning concepts no longer change?

55. Are newer versions getting better because of parameters and data? / Are parameters the same as data?

56. Can't you advance the model while feeding in data at the same time?

57. Does collecting and sending context cost more tokens? / Solutions to the memory problem

58. How is "thinking" mode implemented? / How do you train a model that thinks?

**Layers and Activation Functions (the section we dug deepest into)**

59. What a layer is / Why you multiply the first layer's result again in the next layer

60. What the intermediate values and the final score are / How many weight sets, and who decides

61. Aren't multiple weight sets ultimately collapsing into one value? / Do you just pick the single best set?

62. What an activation function (ReLU) is, and when and why you use it / The meaning of turning negatives into 0

63. Why stacking layers without ReLU collapses them into a single layer (verified with numbers)

64. How bending + stacking layers creates complex patterns

**Moving On to Practice (PyTorch Implementation)**

65. Does the curriculum include everything learned so far? / Is it OK to go top-down with PyTorch?

66. (Day 2) The regression code in PyTorch / How deeply do you need to understand the concept of linear regression?

67. The exact difference between linear regression and classification

68. Why there are multiple evaluation metrics (MAE, RMSE, R²), and what to judge by

**How Layers, Sets, and Features Relate (the second section we dug deepest into)**

69. The principle of getting closer to the answer via matrix multiplication and bending / who created this principle

70. Whether each layer is trained simultaneously, or optimized in order from layer 1

71. Whether having multiple weight sets is separate from layers (horizontal vs. vertical)

72. Where layer 2's sets come from / the rules for the number and size of sets

73. Whether layer 2's sets are fixed at 1, or change dynamically

74. What situation calls for 3 or 4 layers / whether only the last layer is fitted to the output

75. Which layer's weights are used during actual use (inference)

76. Whether, with 5 layers, the answer comes from the layer closest to the correct answer

77. Whether layer 2's weights are layer 1's gradient multiplied in (forward pass vs. backpropagation direction)

78. Whether an intermediate value is one feature, or a layer is one feature

79. How features arise from random weights

80. Whether it's bad for the weight sets to be too similar (initialization)

**Turning Data into Numbers (Feature Encoding)**

81. What one-hot encoding is

82. When you need embeddings

83. Whether changing the features (inputs) means retraining the model

**Digging Deeper into Embeddings**

84. Whether embeddings cluster similar words together in vector space

85. What each individual number inside an embedding vector means (is it a position?)

86. Whether embedding numbers are learned the same way as in deep learning / who sets the vector length

87. Whether you use embeddings for lookup, or as input

88. How you feed an embedding array as input and normalize it

**What Is a Model (a "Frame")? — where the map widened**

89. Whether choosing the right one matters for embedding models too

90. Whether "model" means the algorithm (frame), or taking a finished one and using it

91. How the way weights are made differs by frame / whether loss is an algorithm for measuring

92. Whether trees need layers

**Implementing Forward + Backprop from Scratch in NumPy (hands-on, 2026-07-01)**

> Code: [backprop.py](../code/backprop-numpy/backprop.py) — studied by **filling it in line by line**, answering questions from the AI.

93. How the derivative formulas in backprop work

94. What exactly is a hidden neuron — is it a weight set

95. Why stacking 8 weight sets gives `(8, 2)` (how to read a matrix's shape)

96. Why we do matrix multiplication / does multiplying produce weights faster

97. What bias is and what `b1` means

98. Whether the bias value goes in randomly / what `b1` is at the first run

99. Whether bias is carried along in the forward pass like an input, or is a separate parameter

100. Broadcasting — adding arrays of different shapes / how bias gets added in the forward pass

101. What z (weighted sum) and a (activation output) mean in the forward pass

102. Why we need to make W2 when W1 already has 8 neuron sets

103. Why there's one bias per neuron (8) / why it's not a single fixed `+1`

104. Why b1 has 8 and b2 has 1 / at which stage the bias is added

105. Why the output layer uses sigmoid instead of ReLU (activations differ by layer)

106. What happens to W·b with 3+ layers (extending manually)

107. Do frameworks build layers in a loop instead of one variable at a time

108. Who decides how many hidden neurons per layer, and on what basis (hyperparameters)

109. Is z2 a matrix / what the intermediate raw score means / how a probability comes from it

110. Is it z2 or -z2 that goes into exp / why sigmoid has this shape / does z2=1 mean 100% confidence

111. Is there more than MSE for loss / BCE for classification and why

112. Is backprop one method or many / does a framework's `.backward()` contain many methods

113. What autograd precisely is — does it decide the loss automatically?

114. Are there more losses than MSE·BCE / is it all measure-then-differentiate in the end

115. Do I need to master the algorithms and math proofs of every loss (learning strategy)

116. Is dz2 a gradient / why the 2 (backprop `d`-prefix and layer number)

117. What exactly is the principle of computing a derivative/gradient / is dz2 four gradients (one per row)

118. What transpose `.T` is — swapping rows and columns / why it's used in dW2

119. Why divide dz2 by 4 (N) — because the loss is a mean

120. What exactly is in the loss variable / when is it used

121. Why loss is 1 number but dz2 is 4 / when do samples get combined

122. Is the gradient "the error size" / do we multiply the error onto the weight to move it

---

## 1. Machine Learning (ML) and Deep Learning (DL)

- **Machine learning**: an approach where, instead of coding the rules by hand, you train a model to learn patterns (rules) from data.
  - Traditional programming: *rules + input → output*
  - Machine learning: *input + answers → rules (model)*
- **Deep learning**: a branch of machine learning. It learns by stacking artificial neural networks deep, in many layers.
  - The key difference is **how features are handled**. Traditional ML has humans design the features, whereas DL extracts features on its own as data passes through layers, from low-level (lines, edges) up to high-level (faces, objects).
- Relationship: **AI ⊃ machine learning ⊃ deep learning**. LLMs and transformers are also a form of deep learning.

---

## 2. Why GPUs and Memory Matter in Deep Learning

- The core of deep learning computation is repeated, enormous **matrix multiplication**.
- **GPU = parallel computation**: thousands of simple cores handle independent calculations simultaneously → ideal for matrix multiplication.
- **Why memory (VRAM) matters**: during training, all of the following occupy memory at once.
  - Model parameters (weights)
  - Activations — intermediate results saved for backpropagation
  - Optimizer state and gradients
- If you run short → "CUDA out of memory" → you have to shrink the batch or the model.
- **Speed (compute) vs. capacity (VRAM)**: if only one of the two is good, you get a bottleneck.

---

## 3. Forward Pass and Backpropagation (the cooking analogy)

- **Forward pass**: ingredients (the input) go through several stages and a dish (the result) comes out = producing the answer.
- **Backpropagation**: you taste the finished dish, find "where it went wrong" by working backward, and fix it = passing the degree of error backward, from the end toward the front, to correct the weights.
- Repeat this process thousands to tens of thousands of times and the model gradually gets smarter.

---

## 4. Research Engineers and Understanding Hardware

- Semiconductor **design itself** (circuit design, Verilog, etc.) is a separate field → it's something you'll almost never do directly.
- But a system-level understanding of **"why the hardware behaves the way it does"** is a powerful weapon.
  - Understanding the memory hierarchy, memory-bandwidth bottlenecks (memory-bound), and how parallel processing works → directly tied to optimizing code.
- Priority order: ML/DL understanding + engineering skill > understanding GPU/memory systems > semiconductor design knowledge.
- Recommended direction: rather than "how to design semiconductors," focus on **"how the GPU and memory execute deep learning computations"** (GPU architecture, CUDA, memory optimization).

---

## 5. The Flow of How a GPU Executes Deep Learning Computation

1. **Move the data to the GPU**: disk → CPU memory → VRAM. The CPU↔GPU stretch (PCIe) is slow and becomes a bottleneck → preparing ahead (prefetch) matters.
2. **Pull it on-chip to compute**: VRAM is a "distant warehouse" from the core's point of view. The actual computation happens in the nearby cache and registers. A big matrix is sliced into small pieces (tiles) for processing.
3. **Thousands of cores do matrix multiplication at once**: because the result values are independent of one another, they can be processed in parallel. They're grouped into warps to fill idle time.
- **Key point**: number of cores = "how fast you compute," memory speed/capacity = "how fast and how much data you can supply."

---

## 6. Can You Skip the CPU and Load Straight Onto the GPU?

- The original data actually lives on **disk**: disk → CPU memory → VRAM.
- Reasons for going through the CPU: preprocessing, deserializing the weights, and orchestrating the whole job.
- Techniques to reduce or bypass the CPU:
  - **Pinned memory**: speeds up the copy.
  - **GPUDirect Storage**: transfers disk → GPU directly (bypassing CPU memory).
  - **GPU preprocessing**: doing decoding and augmentation on the GPU.
- Principle: things you use continuously, like model weights, get **loaded into VRAM once and left there to the end**.

---

## 7. VRAM · Cache · Register · Core — The Physical Layout

- **VRAM**: a **memory chip separate from the GPU chip**. It's soldered onto the graphics-card board next to the GPU. The types are GDDR / HBM. Large capacity, but a "distant warehouse" from the core's point of view.
- **GPU die**: a single piece of silicon. Everything below is **etched into it as circuitry** (not separate parts).
  - **Core**: performs the actual multiplications and additions. Not a separate chip, but a region on the die.
  - **SM (Streaming Multiprocessor)**: a bundle of dozens of cores. There are tens to hundreds of these inside the die.
  - **Register**: right next to the core, the fastest and smallest. Holds the value currently being worked on.
  - **L1 cache**: dedicated to each SM. / **L2 cache**: shared by all SMs.
- **Distance = performance hierarchy**: the farther away, the larger and slower (VRAM); the closer, the smaller and faster (register).

---

## 8. How Data Travels to the Core

- It's not "passing through one stage at a time in order," but rather **"when something is needed, it's fetched starting from the closest place."**
  - Core requests → check register/L1 → if not there, L2 → if not there, VRAM.
  - Once fetched, data is left in the cache for reuse.
- **The register is not the next step after the cache**: L2→L1 is automatic (caching), whereas the register holds "the value the core has in hand right now" and is loaded up by an explicit instruction.

---

## 9. How Does a Core Decide What It Needs?

- The core **doesn't decide**. It just executes machine-code instructions one by one, exactly as given.
- Each instruction directly specifies the data (address) it operates on. E.g., `load the value at address A into R1`.
- The instructions come from the **source code** → the compiler translates them into machine code.
- **A subtle point**: addresses are often not fixed values but computed during execution, like `the i-th element of an array`.
  - On a GPU, thousands of cores execute the **same instruction**, but **each computes a different i using its own unique number (thread ID)** → same code, different data → parallel processing.

---

## 10. When the Data Is Bigger Than VRAM (e.g., 100GB of data vs. 24GB of VRAM)

- You don't load all 100GB at once. The original stays on disk, and in VRAM you keep:
  - model weights (resident) + the current batch (a small piece) + activations/workspace.
- **It's streamed in batches**: load batch 1 → process → clear → load batch 2 → repeat.
- **What clears it is the code (framework), not the core**: it's not "because the core finished" but "the moment the program no longer references that data" that memory is reclaimed and reused (e.g., PyTorch's caching allocator).
- The next batch is prepared in advance by the CPU (prefetch) while the GPU is still computing.
- **The fork in the road**:
  - The dataset is 100GB → the easy case. Solved with batch streaming.
  - The model itself is 100GB → the hard case. You need all of it for every computation → you need model parallelism / offloading / quantization.

---

## 11. Does the Model Go on the GPU? (training vs. inference)

- As long as you compute on the GPU, the model (= a bundle of weight numbers) is **almost always resident in VRAM**.
- "Model" is not a training-only concept. **The same model goes through two phases**:
  - **Training**: adjusts the weights from random toward the right answers. The heaviest (model + data + gradients + optimizer state).
  - **Inference**: with the weights fixed, it produces answers for new inputs. Relatively light (only the model is resident; only the input is fed in).
- Analogy: **the model = the factory machine (bolted down), the data = the raw material passing through (the conveyor belt).**

---

## 12. The GPU's Two Uses

1. **Training**: the phase that creates the weights. "Adjusting/learning" is more accurate than "measuring" — you're not measuring the answer but creating it.
2. **Inference**: the phase that uses the created weights to produce an answer for an input = actual use.
- Order: complete the weights through training → freeze them → use them repeatedly via inference.
- Analogy: training = finishing the recipe, inference = cooking with that recipe every time.

---

## 13. What VRAM Really Is

- **VRAM = Video RAM** (the V is not Virtual but **Video**). It comes from the GPU's original role (driving the display).
- Differences from regular RAM:
  - **Who/where uses it**: regular RAM = for the CPU (motherboard), VRAM = for the GPU (graphics card).
  - **Bandwidth**: VRAM's is far larger — designed to supply data to thousands of cores all at once.
  - **Standard**: regular RAM = DDR, VRAM = GDDR/HBM.
  - **Expandability**: regular RAM can have modules added, VRAM is soldered down → you can't add more later (which makes it a purchasing factor).

---

## 14. Is Weight Training All Matrix Multiplication?

- Matrix multiplication is **the central axis and the heaviest part (often 90%+)**, but it's not everything.
- Non-multiplication operations that run alongside it:
  - **Activation functions** (ReLU, sigmoid, etc.): they introduce nonlinearity. Without them, stacking layers deep becomes meaningless.
  - **Addition and normalization** (bias, normalization).
  - **Loss computation and differentiation** (computing gradients in backpropagation).
  - **Weight updates** (the optimizer, e.g., Adam).
- Why matrix multiplication gets emphasized: it overwhelmingly dominates time and resources (the GPU's tensor compute units are specialized for it).

---

## 15. The Data Flow of an Image Classification Model (dog vs. cat)

1. Preparation: image files + the answer for each image. They exist on disk/CPU.
2. Preprocessing (CPU): convert each image into a **numeric array (width × height × channels)**. Includes resizing and normalization.
3. The answer also becomes a number (a label): dog = 0, cat = 1.
4. Copy to the GPU: the image (numbers) and the answer (number) go to VRAM **paired together**.
5. On the GPU: predict (forward pass) → compare with the answer (loss) → correct the weights (backpropagation), and repeat.
- **The answer label has to be on the GPU alongside it** so you can compute the degree of error.

---

## 16. The Preprocessing Format and "What It Judges By"

- "Width × height × channels" is just **the format of the container that holds the data**. The number of channels varies with the data (grayscale 1, color 3, special imagery even more).
- The model **doesn't just look at color — it looks primarily at shape**.
  - The secret is **the relationships between neighboring pixels**: a change in brightness → an edge → corners, curves, contours.
  - A CNN scans the image with small windows (filters), detecting the basic building blocks of shape.
  - As it passes through layers: lines/edges → parts (ears, nose) → larger patterns (a cat-ish face).
- Color is also used as a supporting clue, but over-relying on color causes errors (e.g., calling a brown cat a dog).

---

## 17. Where Does the CNN Model Come From, and Who Sets the Criteria?

- **Architecture**: you often take a proven one (ResNet, EfficientNet, etc.) and use it (designing your own is also possible).
- **Two paths for how you train**:
  - Take just the architecture and train it from scratch on your data.
  - Take the already-trained weights as well and refine them on your data = **transfer learning / fine-tuning** (common in practice).
- **Who sets the criterion for pixel color difference / shape?** → **A human doesn't set it. The model learns it on its own from the data.**
  - The criterion is **the numbers inside the filters = the weights**.
  - They start random → get it wrong → compare with the answer (loss) → backpropagation finely adjusts the filter values → repeat → they settle into things like a "vertical-line-detecting filter."
  - **The only basis = "the direction that gets more answers right."** Humans provide the goal (the answers), the architecture (CNN), and the data; the model figures out what clues to look at.

---

## 18. How You Tell the Model "Learn to Classify Dogs/Cats"

- You don't explain it in words — you **attach an answer number to each image and pair them up** (labels). Automatic labeling via folder structure (dog/, cat/) is also possible.
- The answer labels = **the grading key**: the model compares its prediction with the label, grades it, and corrects itself by however much it was wrong.
- **The type of task is determined by the form of the answer**:
  - Answer is a category (0/1) → **classification**, with a classification loss (cross-entropy, etc.).
  - Answer is a continuous number (price, etc.) → **regression**.

---

## 19. There Is Also Learning Without Labels

| Approach | Answer key | Core idea | Example |
|---|---|---|---|
| **Supervised learning** | Humans assign labels | Learns by looking at the answers | Dog/cat classification |
| **Unsupervised learning** | None | Finds only the patterns/structure within the data | Clustering, anomaly detection |
| **Self-supervised learning** | The data generates it itself | Extracts the answer from the data itself | An LLM's "guess the next word" |
| **Reinforcement learning** | None (replaced by reward) | Maximize reward, trial and error | AlphaGo, robot locomotion |

- Study analogy: supervised = a workbook with an answer key / unsupervised = finding patterns by looking at materials without answers / self-supervised = covering the book and guessing what comes next (the book is the answer) / reinforcement = repeating a game while only getting score feedback.
- **Self-supervised learning** in particular lets the data create its own answers without humans labeling → it can learn from internet-scale data → which is the heart of today's large language models.

---

## 20. What Is a Gradient?

- **A value that tells you, for each weight, in which direction (sign) and by how much (magnitude) to change it so the loss decreases** = a direction signal.
- It's the slope under your feet for descending to the lowest valley on the "hill" of error (loss).
- **Backpropagation** computes this gradient, and the **optimizer** (Adam, etc.) looks at it and corrects the weights one step at a time.
- One reason training eats so much memory: you have to store a gradient value separately for every weight.

## 21. A Weight Is Not the "Probability of Being the Right Answer"

- **Probability (e.g., 70% dog) = the final result of the computation.** **A weight = a part you multiply in along the way to produce that result.**
- A weight = **an importance knob** that sets "how important each piece of information should be."
  - E.g., `judgment score = (ear shape × big weight) + (nose shape × medium) + (fur color × small weight)`
- Training = the process of turning these knobs to fit the answer. The final output that comes out of it is the probability.

## 22. The Model Finds the Weights on Its Own (but it's "automatic adjustment," not "judgment")

- The weight values aren't set by humans — they **settle into place on their own through training**. But it's not clever judgment; it's **mechanical adjustment**.
  - Start random → get it wrong → the gradient points the direction → move a little → repeat → it rolls into a good position by itself.
- What humans set: the model's **architecture**, the **answers (labels)**, the **loss function (grading method)**, and settings like the **learning rate**.
- What values to fill the weights with is left for the model to discover from the data.

## 23. Matrix Multiplication Isn't a Tool for "Finding" Weights — It's a Tool for "Using" Them

- Matrix multiplication = the **prediction (forward pass)** step that handles `input × weight` as one bundle. It doesn't find the weights directly.
- The principle of finding the weights = a repetition of three steps:
  1. **Predict** (matrix multiplication, forward pass)
  2. **Measure how wrong it is** (loss)
  3. **Compute which weights to fix and how** (gradient, backpropagation) → adjust the weights
- Matrix multiplication enters as the **tool** for ① (prediction) and for computing the gradient. "Deep learning = matrix multiplication" means that's the tool of the computation — not that the principle of finding weights is itself multiplication.

## 24. On What Basis Does Backpropagation Issue Its Correction Instructions?

- **There's just one basis: the direction in which the loss (the degree of error) decreases.**
- It figures out the direction and magnitude in one shot through **differentiation (computation), not by plugging in values one at a time to test.**
- **Why it's called "back(ward)"**: after computing "how wrong it was" at the output, it passes that responsibility backward — output → earlier layer → … → input — computing the gradient for each weight along the way (the chain rule). One sweep backward yields the correction instruction for every weight all at once.

## 25. The Principle of How Differentiation Knows the "Effect on the Loss" (without math)

- The core in one sentence: **"If I turn this knob (weight) just a tiny bit, which way and by how much does the result (loss) change?"**
  - It's like turning a faucet slightly to see whether the water gets hotter.
  - If turning it slightly makes the loss go down → go that way. If it drops sharply → the effect is large. If it barely changes → the effect is small.
- But **without actually turning each one**, it knows the change in advance just from how "tilted" the formula is (like knowing which way a ball will roll on a slope without rolling it). → It can handle millions of them at once.
- Whether the effect is "large or small" depends on the surrounding values: in `result = input × weight`, if the input is large then the weight's effect is large too.

## 26. How Do You Confirm Things "Got Better" Using the Gradient?

- The **gradient (from differentiation) is only the prediction that "this direction will be better"** — it's not yet actual confirmation.
- You **move a little in that direction → on the next lap, predict and grade again with the new weights (forward pass + loss)** → at that point you **confirm** whether the loss actually went down.
  - E.g., loss 80 → (move as predicted) → grade again 75 → move → 71 …
- **Why you move only "a little"**: differentiation can only be trusted for the slope at "the current spot." Jumping far in one go can change the terrain and make things worse. This step size is the **learning rate**.
- "We don't test one at a time" (finding the direction) and "we take one whole step and then grade once to confirm" are not contradictory.

## 27. The Basis of Differentiation's Judgment, and the Origin of the Input and Weight Values

- **Basis of judgment**: differentiation only **computes the fact** that "if you increase this weight, the loss changes by this much." The goal "let's reduce the loss" is what sets the direction (gradient descent).
- **Input values**: the **data** we provided (fixed). They're not set, they're given.
- **Weight values**: only the very first time are they random; after that, the **current value adjusted on the previous lap** is used as-is in the next round of differentiation. → When the position changes, the slope (gradient) changes each time too.

## 28. The Essence of Deep Learning

- **Deep learning (training) = the process of adjusting weights one step at a time, in the direction that reduces the loss, toward the "criterion for getting answers right" that the data has defined — finding a value that is good enough.**
- Two asterisks:
  - It's closer to **"good enough"** than to "the absolute best" (it sometimes settles in a reasonably low spot rather than the perfect minimum).
  - **What counts as a "good weight" is decided by the data and the answers.** With the same architecture, the result changes depending on what data you give it.
- The model architecture, the GPU, matrix multiplication, and differentiation are all **tools that make this process possible and fast.**

## 29. The Precise Meaning of "The Cores Keep Running Matrix Multiplication"

- It's true that the inputs and weights are prepared as matrices (tables) and the cores process the multiplication.
- But **one core doesn't run the whole matrix — thousands of cores compute different cells of the result simultaneously** (parallel).
- Across the whole training, it does repeat, but **it's not running the same thing each time — it multiplies again with slightly adjusted weights.**
- And it doesn't only run multiplication (activation functions, loss, and gradient computation are mixed in too). It's just that multiplication takes up most of the time, hence the phrasing "the cores run matrix multiplication."

## 30. Does More Iteration Always Mean More Accuracy? — No

- **Early on**: the more you run it, the more the loss drops and things improve.
- **Convergence**: as you near the bottom of the valley, running it more barely changes anything (a signal that it's "sufficiently trained").
- **Overfitting**: run it too long and it **memorizes the training data wholesale** → it does great on the training data but actually does worse on data it has never seen.
- Exam-prep analogy: if you memorize the answers to past exam questions wholesale, you'll ace those past questions but get the actual exam (new questions) wrong.
- The goal isn't "as much as possible" but **"stopping at the point where it does best on data it has never seen."**

## 31. The Basis for Deciding How Much to Run It

- Split the data into three: **train** / **validation** / **test**.
  - Validation data = not used for training; it's the yardstick for judging "when to stop" (a mock exam).
- **Criterion ① Early stopping**: stop when the validation score stops improving or starts dropping (the most basic).
- **Criterion ② The loss curve**: plot the training loss and validation loss together and stop at the point where the two lines diverge.
- **Criterion ③ Time/cost**: consider value-for-money, like the hourly cost of the GPU.
- "A lot vs. a moderate amount" isn't a fixed value — **it depends on the amount of data and the model size, and you adjust by checking the validation score.**

## 32. What Are the Inputs for Unsupervised and Self-Supervised Learning? — In all three, the data itself

- The difference is not the **input** but **"where you get the answer (the loss criterion) from."**

| Approach | Input | Answer (loss criterion) |
|---|---|---|
| Supervised | Data | Labels attached by humans |
| Self-supervised | Data (part of it hidden) | The hidden part (the data provides it itself) |
| Unsupervised | Data | No answer → replaced by "is the structure well organized?" |

- **Self-supervised example**: "The weather today is really ___" (input) → "nice" (answer, the word that was already in the original text). Humans don't label it.
- **Unsupervised example**: clustering uses "are the same groups close together and different groups far apart?" as its criterion. An autoencoder uses "did it reconstruct the input as-is?" (the input itself is the answer).
- **What they have in common**: the **learning engine** — `input × weight → prediction → loss → gradient → adjust → repeat` — is identical in all three. The only thing that changes is **how the loss is defined.**

---

## 33. Why Can't You Just "Throw Any Number at It and Test"? (combinatorial explosion)

- When there are few weights, throwing numbers (random search) actually works. The problem is the **count**.
- The weights are independent of each other, so you'd have to consider **every combination** → the number of cases grows by **multiplication, not addition.**
  - Outfit-coordination analogy: 10 tops × 10 bottoms = 100 combinations (not 20). Whether they match is determined by the "combination," so you multiply.
  - 1 weight = 10 tries → 2 weights = 100 → 3 = 1,000 → 10 = 10 billion → …
- Real models have **millions to billions of weights** → the combinations exceed the number of atoms in the universe = **the curse of dimensionality.**
- Differentiation finds the direction for **every weight in one backpropagation pass**, all at once, without throwing anything → it's the only realistic way to avoid this explosion.

## 34. Weights Aren't About Comparing Candidates — It's About "Moving a Single Value"

- A common misconception: "you hold several weight candidates and pick the best-fitting combination" → **no** (that's throwing numbers).
- In the differentiation approach, each weight **always has just one current value.**
  - Compute with that single current value (matrix multiplication) → loss → differentiation points the direction → **overwrite the value** (0.30 → 0.28) → discard the old value.
- It does **not** "compute including all the weights that have come up so far." It always computes **with just the single latest value.**
- The "best-fitting combination" isn't chosen from candidates — it's **the final position a single point reaches by moving along the slope.**
  - Mountain analogy: scattering millions of people and comparing (throwing numbers) ❌ / one person walking down step by step along the slope (differentiation) ✅. That one person's current position = the current weights.

## 35. Differentiation's Answer Is a "Direction," Not a "Value" / Line Search

- The output of differentiation is **not** a new value (0.27) but a **movement instruction**: "move this way, by this much."
  - current value + (direction × step size) = new value. This **step size = the learning rate.**
- KIM's "I'll just throw 0.27, 0.29, 0.31 all at it" = **set the direction with differentiation, then try several step sizes and pick the best one.**
  - This is a real method = **line search**. The idea is correct.
- But it's rarely used in deep learning: for each candidate step, you'd have to **recompute the loss of the entire model**, so the cost multiplies by the number of candidates. A single loss computation is very heavy (billions of multiplications).
- Instead, you **take one step at a pre-set stride, then recompute the direction on the next lap.** Each step may not be optimal, but it's corrected often enough to reach the valley well.
- The step size (learning rate) is sometimes set by throwing numbers, **since there are few of them (usually one)** (try 0.01, 0.001, and pick).
- In summary: KIM's intuition was right; the place it applies was just "step size" rather than "weights."

## 36. Is It Ultimately a Matter of Time and Cost? — Yes, but it splits into two kinds

- **(A) Expensive but possible cost** = a pure value-for-money judgment (time/cost). Spend more, and it might get a bit better.
  - How long to run training, how many step sizes to try, when to stop.
- **(B) The realm that's impossible no matter how much money you spend** = you have to change the method itself.
  - Finding millions of weights by throwing numbers → even with a million GPUs running for the age of the universe, you couldn't finish.
- Analogy: walking to Busan (A — slow, but you arrive) and walking to the Moon (B — you'll never get there) are different problems.
- So differentiation isn't simply **"cheaper"** — it's the **key that made possible what throwing numbers never could.**

## 37. Is Math (Calculus, Linear Algebra) Essential for Deep Learning? — It depends on the stage

- **① The using stage** (leveraging existing models, training with libraries): deep math is mostly unnecessary. Differentiation is handled automatically by the tools. **Conceptual intuition** is enough.
- **② The handling-it-properly stage** (diagnosing, changing architectures, reading papers): intuition-level math is a powerful weapon.
  - Calculus = the intuition of "if I nudge it slightly, how does the result change" / linear algebra = a feel for "matrix multiplication and tensor representation" (used the most often) / probability & statistics = "probability, loss, distributions."
  - The intuition to understand the concepts matters more than the ability to crunch the calculations.
- **③ The research / building-new stage**: solid math is required.
- **Recommended way to study**: not "perfect the math first" but **filling in the math at the exact point where you get stuck while working on deep learning.** Math learned after a need arises sticks best. (This session itself is an example of that approach.)

---

## 38. How Does the Model Know to "Look at the Ears"? — It infers it on its own from the answers

- The only answer we gave is **"the whole photo is a cat (1)."** We never taught it "look at the ears" or "the pointy thing is the ear."
- And yet it gains the ability to respond to ear shapes, because: **a filter that responds to "a pattern common to cat photos but absent in dog ones" gets more answers right** → the gradient naturally drags the filter in that direction.
- In other words, it's not a command to "look at the ears" — it's that **the result "responding to such clues turns out to be advantageous for getting answers right"** pushes the model that way.
- **The twist: the model doesn't know the concept of "ear."** It just learns the **numerical association** that "this brightness-change pattern at this location → higher cat probability." The interpretation "ear" is something humans tack on afterward.
- A side effect: it grabs anything that co-occurs with the answer → if the data is biased, it might learn the **wrong clue** (e.g., "indoor background = cat"). This is why good, sufficient data matters.

## 39. Can You Know "What a Finished Model Judges By"? — The black-box problem

- **It's hard to know clearly.** The millions to billions of weights inside the model are tangled together to produce an answer, so it's hard for a human to read off "which number, on what basis" → the **black-box problem.**
- The cause: we didn't write the rules ourselves; the model carved out the patterns on its own. It doesn't translate cleanly into human language. **In a sense, we traded away some explainability for the performance we gained.**
- Still, there are techniques to peer inside (explainable AI, XAI):
  - **Heatmaps**: color the image regions that contributed most to the judgment → indirectly check "did it look at the ears, or the background?"
  - **Filter visualization**: draw out what patterns a particular filter responds to.
  - But these tell you "roughly where it's looking," not "exactly why" — that can't be fully explained.
- Why it matters: in things like medical diagnosis or loan screening, **if you don't know the basis, it's hard to trust and delegate** → the "performance vs. explainability" trade-off is a frontier topic in current AI research.
- Analogy: a genius chef who produces incredible flavors but can't write down the recipe in words. The result is excellent, but the process can't be explained.

---

## 40. One Training Lap Run by Hand (the flower-classification example)

A record of taking what I learned abstractly today and computing it myself with small numbers.

**The rule for the number of weights**
- The number of weights one measurement (input) has = **the number of results (outputs) to produce.**
- Total number of weights = **number of measurements × number of outputs.**
  - 3 measurements, 1 output → 3 weights (1 per measurement).
  - 3 measurements, 3 outputs (3 species) → **9 weights** (3 per measurement, 3 sets, one per output).
- Note: with 3 outputs, **the result values (scores) are 3, but the weights are 9.** (The 9 is the number of weights, not the number of results.)

**Forward pass (making a prediction)** — matrix multiplication
- `prediction = (input1 × weight1) + (input2 × weight2) + …`
- E.g., input `[2,3,1]`, weights `[0.5,1.0,2.0]` → (2×0.5)+(3×1.0)+(1×2.0) = **6**.
- In the matrices, A = input (horizontal/rows), B = weights (vertical/columns). Even for the same numbers, the row/column orientation is distinguished (to line up the multiplication pairs).

**Loss (the degree of error)**
- The difference (the gap) is the core: prediction 6, answer 10 → difference 4.
- In practice you **square** it: `loss = (prediction − answer)² = (−4)² = 16`.
  - Why square: ① removes the sign (turns ± into +) ② penalizes large errors heavily ③ **the derivative is clean.** It's adopted because "the property suits learning well" more than for efficiency (speed).
  - The loss method is chosen from several options (MSE, absolute value, cross-entropy, etc.). A human picks the one that fits the problem.

**Direction (matching the prediction to the answer)**
- prediction 6 < answer 10 → need to **increase** the prediction to lower the loss → the direction is to increase the weights.
- Check: increasing the weights (0.5,1.0,2.0)→(0.6,1.2,2.2) makes the prediction 6→7 and the loss 16→9. It really does decrease.

**The origin of the answer vs. the prediction (important)**
- **Answer**: the fixed label attached to the data (the value we gave). It never changes throughout training. (the archery target)
- **Prediction**: the value the model computes on the fly from input × weight. It changes when the weights change. (the arrow you shoot)
- Training = matching the moving prediction to the fixed answer.

## 41. Normalization — making the sizes of measurements fair

- **The problem (KIM found it)**: when a measurement's number is large, that weight's influence grows too. But this can be **fake influence caused by units** (e.g., measured in mm it's 50, in cm it's 0.3).
- **The fix**: before training, scale all measurement items to a similar range (e.g., 0–1) = normalization.
- **Division of roles**:
  - The size of the measurement → made fair by normalization (removing fake influence).
  - The real importance → handled by the **weight** (set by training).
- **How the value is determined (not human intuition — automatic)**: plug "where the current value sits within that item's overall min–max range" into a formula.
  - `(value − min) ÷ (max − min)`. E.g., for a stem-count range of 1–6, a value of 2 → (2−1)÷(6−1) = **0.2**.
  - It doesn't matter whether it's a count or a length. Each item is spread to 0–1 within its own range, so the units are erased and only the relative position remains.
- **Tools (libraries)**: scikit-learn's `MinMaxScaler` (spread to 0–1), `StandardScaler` (mean 0, spread 1; the most common in practice). `fit` (learn the range) → `transform` (convert). The same pattern as a model's `fit`/`predict`.
- **A practical caution**: `fit` only **on the training data.** For test data, apply only `transform` (looking at the test data ahead of time would be "peeking").

## 42. What the Result of Differentiation Really Is

- If there are 9 weights, then the **differentiation result is also 9** (one number per weight). This bundle = the **gradient.**
- Each number carries two pieces of information:
  - **Sign (+/−)** = direction. Negative means "increasing it lowers the loss → increase it"; positive means "decreasing it lowers the loss → decrease it."
  - **Magnitude** = influence. Large → fix it a lot; small → fix it a little.
- Why the value differs per weight: because the **input multiplied alongside it differs** (a weight in a position with a large input has a large effect).
- **One weight's gradient = "that weight → effect on the score (= the adjacent input value)" × "the score → effect on the loss"** (the chain rule). Backpropagation computes all 9 at once, working from the back.
- The differentiation result is **not multiplied into the weights** — you **update the weights by adding or subtracting in that direction** (e.g., 0.5 → 0.52).

## 43. Learning Rate — "How big a move to make" is set by humans

- You **don't move the full gradient value as-is.** You multiply by a certain ratio and move only a little. That ratio = the **learning rate.**
- It splits into two parts:
  - **The ratio among the weights** (who to increase by how many times more) = **set exactly by the gradient (automatic).** E.g., gradients −6 vs −2 → exactly 3×.
  - **How big the overall move** is = the **learning rate (chosen in advance by a human).** Multiplying by the learning rate still preserves the ratio between weights (3×).
- E.g., learning rate 0.01 → gradient −6 moves by 0.06, −2 moves by 0.02.
- The learning rate isn't automatic; it's a setting a human picks by trying 0.1 / 0.01 / 0.001, etc.
  - Too large: it leaps over the valley and becomes unstable. Too small: it's too slow. One of the most important settings.
- Analogy: differentiation = the slope of the terrain (automatic), learning rate = the stride you take (set by a human).

---

## 44. LLMs Also Have an Answer — the loss in self-supervised learning

- An LLM's answer isn't assigned by humans — it's **"the very next word in the text."** The next word that was already in the original text is the answer.
  - input "The weather today is really" → answer "nice" (the word that was in the original text).
- The loss computation is identical to the flower example: the model assigns a probability to every word, and it's graded by **whether it gave a high enough probability to the correct word** (cross-entropy).
- So an LLM = "ultra-massive fill-in-the-blank." Because you can get the answers from text **for free, infinitely**, you can train on the entire internet → this is the heart of what makes LLMs smart.
- It only did next-word prediction, yet it picks up grammar, context, common sense, and even reasoning along the way (because getting it right requires understanding the meaning).

## 45. Turning Text into Numbers — tokenization and embeddings

- It goes through two stages:
  1. **Tokenization**: cut the text into pieces (tokens) and number them. "The weather today is nice" → [1024, 5847, 392].
  2. **Embedding**: turn those numbers back into **bundles of numbers (vectors) that carry meaning.** "King" and "queen" get similar numbers; "chair" gets entirely different ones.
- Why numbers alone aren't enough: a number (1024) is just a name tag — its size and order carry no meaning, so it can't capture the meaning-relationships between words.
- **The dictionary (vocabulary)**: before training, it scans massive amounts of text to **automatically** build a list of pieces + a number table, then freezes it. Humans don't set it by hand. It splits words into pieces (subwords) so it can handle even words it has never seen.
- **The principle behind how embeddings cluster**: "words that appear in similar contexts have similar meanings." As it learns next-word prediction, the embeddings of similar words **become similar on their own** (because embeddings are learned weights too). It's not told to group them directly — it's a byproduct of getting answers right.
- A remarkable result: meaning-relationships even show up as directions (e.g., `king − man + woman ≈ queen`).
- **An embedding isn't a model — it's the model's first component (layer).** Its substance is a "number → meaning-number-bundle" table, and the numbers in that table are learned weights too. They're sometimes detached and used separately for things like meaning comparison.

## 46. The Training Loop Is Identical Everywhere — Images, Text, LLMs

- **What doesn't change (the engine)**: forward pass (prediction) → loss → gradient via differentiation → adjust weights by the learning rate → repeat. Common to every field.
- **The 3 things that change by field** (all of it concerns what happens before and after feeding the engine):
  1. **How you turn the input into numbers** (image = pixels, text = tokens/embeddings, video = images + time)
  2. **How you define the answer (loss)** (label / next word / how well structure is organized)
  3. **The model architecture** (image = CNN, text = transformer)
- So once you truly understand one (the flower example), a new field just means "looking afresh at only 3 things: the input conversion, the answer, and the architecture."

## 47. Parameters ≠ Data

- **Parameters = weights** (the very thing learned today). "70 billion parameters" = 70 billion weights adjusted by training. It's the flower example's 9 scaled up to 70 billion.
- **Data** is the raw material given from outside (internet text, etc.); **parameters** are the numbers created inside the model through training.
- When training ends, **the data disappears and only the parameters remain.** The internet text isn't sitting whole inside the model — the result of training is compressed into the parameters.
- Analogy: data = the books you read, parameters = the knowledge and connections in your brain. When you take the exam, you need only your brain (parameters), not the books (data).

## 48. The Model Doesn't Learn While It's Being Used (on purpose)

- Technically possible, but the two are deliberately separated (training vs. inference). A user's question is inference, and at that point **the weights are fixed.**
- 3 reasons it isn't done: ① cost (backpropagation every time is far too heavy) ② risk (a single piece of bad data could ruin the model) ③ loss of control (losing consistency and traceability).
- Instead, interactions are **collected as data, vetted, and then used to train the next version in a controlled environment** → it advances in version-sized increments.
- To make it *look* like it has advanced without changing the weights: **conversation memory** (continually including context in the input) and **referring to external material** (putting relevant documents into the input).

## 49. Context and Token Cost

- The model **can't remember** the previous conversation (the weights are fixed). So with each request, the **entire conversation so far is put into the input again.**
- The result: the longer the conversation, the more **tokens pile up and the cost grows** (billed by token count). A single call reprocesses the whole context, so even if only the last part changes, the cost is based on the whole.
- Once you exceed the **context window** (the token limit it can take at once), the oldest conversation gets cut first → the "forgetting what was said earlier" phenomenon.
- A side effect: even when old information is still in the input, the model pays less and less attention to it (attention decay based on position).
- How to manage it: **summarizing** (shortening old conversation) and **selecting** (including only the parts relevant to the current question). The heart of designing LLM apps = managing "what to put in and what to leave out."

## 50. Solutions to the Memory Problem (under active development, as of 2026)

- The key insight: **"the context window is not storage — it's working memory (RAM)."** Things to remember long-term go in a separate store.
- Three directions:
  1. **External memory**: store the conversation outside (in a vector DB, etc.) and **pull out only what's needed.** Already in practical use — there are cases of cutting tokens 4× while *improving* accuracy (less stuffed in, less confusion).
  2. **OS-style tiered memory**: treat context = RAM and external = disk, swapping information (applying exactly the hardware memory hierarchy learned today).
  3. **Architectural improvements / test-time learning**: make attention more efficient, or use the context like training data to learn at test time (a hoped-for breakthrough in 2026).
- A side effect: external memory comes with a new security threat called **"memory poisoning"** (an attack that plants bad data into the memory).

## 51. Why Recent Models Are Getting Better — Not "bigger" but "smarter"

- The old formula (more parameters + more data + more compute) is one axis, but it has hit **diminishing returns.** The effect of "just scale it up" is shrinking.
- The several levers behind recent performance gains (acting at once):
  1. **Test-time compute**: making it "think" longer before answering. The biggest driver lately. A small model + thinking more sometimes beats a large model.
  2. **Post-training reasoning** (RLHF, reinforcement learning, etc.): making the same model smarter.
  3. **Data quality + architectural improvements** (e.g., Mixture of Experts).
  4. **Surrounding tools and systems** (combining search, tool use, etc.) → better perceived performance.
- Analogy: the shift is toward **using the brain you already have better**, rather than growing the brain bigger.

## 52. How Do You Make a "Thinking" Model?

- What "thinking longer" really is: **generating more intermediate working-out tokens** before answering (chain-of-thought). It's nothing mysterious — just producing more tokens.
- PyTorch handles only "predict one token." "Thinking longer" is a strategy on top of that:
  - **Guiding it via the prompt** ("think step by step") / **calling it multiple times and picking** / **training it to do so.**
- **A serious thinking model is a different model in itself** — a model **further trained to think well**, combined with the strategy that runs it.
- **The training principle (the core)**: you don't grade the working-out process against an answer — you **only reward "whether the final answer was correct"** (reinforcement learning, RLVR).
  - prediction = the entire working-out + the final answer / answer = only the final answer / grading = wholesale encouragement of the working-out that got the answer right.
  - Even without teaching it "think like this," the goal of "getting it right" drags the model toward thinking at length on its own (the same principle as the dog/cat model coming to look at the ears without being instructed to).
  - It even discovers problem-solving methods no human taught it (like self-checking) → more powerful than "imitation."

---

## 53. What Is a Layer?

- **A layer = a "computation step" a measurement passes through on its way to the final result.** One computation = 1 layer, two = 2 layers (hence "deep").
- What each layer does is the same: **multiply input × weights and sum it all.** Only what goes into the "input" slot differs.
  - Layer 1: takes the measurements → produces **intermediate values.**
  - Layer 2: takes those intermediate values → produces the next intermediate values (or the final score). **From layer 2 on, the input is not the measurements but the previous layer's intermediate values.**
- **Final score = the last layer's result = the model's answer** ("how likely it is to be this species"). Intermediate values aren't the answer; they're material to pass to the next layer.
- "Layer" (position/space) and "one training lap" (forward pass → loss → backpropagation, time/iteration) are on **different axes.** One lap sweeps through all the layers, forward and backward, once each.

## 54. Weight Sets and the Number of Intermediate Values

- A layer doesn't make just 1 result — it can make **several. N weight sets → N intermediate values.**
  - Multiplying the same input by **different weight sets** gives different results. E.g., [2,3,1] × set A → 6, × set B → 4 → intermediate values [6,4].
- The number of sets is a **setting a human chooses (a hyperparameter).** There's no formula; you tune it by experiment (like the learning rate).
- **The sets don't converge to the same value.** They diverge and learn to each handle **a different feature** (if they looked at the same thing, the information would be redundant — a loss for getting answers right). Like several interviewers each assessing skill, character, and experience.
- The sets aren't **candidates to choose from — they're teammates who collaborate** → all of them are passed to the next layer and used together (picking just one would throw away the other features).

## 55. Activation Functions (ReLU) — Why They're Needed (the concept we dug deepest into today)

**What ReLU is**: a simple rule that takes one number and outputs one number = **"if negative, 0; if positive, leave it."** (3→3, -8→0, 12→12). It's unrelated to weights, and it operates between layers **during the prediction computation (the forward pass).**

**Where it runs**: right after a layer finishes its weight multiplication, just before the intermediate values pass to the next layer.
- E.g., layer 1's result [6, -4] → ReLU → [6, 0] → on to layer 2. (Only the negative -4 becomes 0; the positive stays.)

**What if it becomes 0?**: that value can't have any effect in the next layer, since `0 × weight = 0` (abstaining from the vote). But **it's not permanently discarded** — when a different input comes, it can come out positive and switch back on. It's a **switch** that turns on and off depending on the input. And the 0 is 0 **only in that layer**; in the next layer it mixes with other values to become a new value (not fixed).

**Why it's needed — without bending, the layers collapse into one** (verified with numbers):
- measurements [2,3], set A = [1,2] · set B = [3,1] → layer 1 [8,9] → layer 2 weights [2,1] → final 25.
- But this 25 is **identical to a single layer that multiplies the measurements by [5,5]**: (2×5)+(3×5)=25.
- That is, **if you stack only multiplication without bending, then no matter how many layers, it all reduces to a single layer of "measurement × some number."** The point of stacking layers becomes 0.
- ReLU's "bend" prevents this collapse → each set independently finds its own feature. **The essence isn't "making it positive" but "bending it once,"** and that bend forks the paths apart.

**The principle of how bending + stacking layers creates complex patterns**:
- Layer 1 + ReLU = creates a bent line (a simple piece). The next layer **combines** those bent lines, increasing the number of bends, gradually forming a more complex boundary.
- Even jumbled data that can't be split by a single straight line (O X O X...) can be split by a bumpy boundary, by stacking and combining many bent lines layer by layer → "line → curve → ear → face."
- But this part is the result of hundreds of bends overlapping, so **tracing the numbers by hand is impossible.** The intuition "combining bends makes it complex" is enough; to actually see it, the right move is to **add layers in code and watch the boundary grow more complex** with your own eyes.

**Summary of why you stack layers**: it's not "to find the weights more accurately" (every model trains, regardless of the number of layers). It's **to express more complex patterns than a single layer can** (more expressive power). The accuracy gain is the result of that.

---

## 56. Switching to Practice — Implementing It Top-Down in PyTorch

- The curriculum (Phase 1–6) covers nearly everything learned so far, and more. "Day" isn't a calendar date but an **order.** One file can take several days; that's fine.
- **Why go to PyTorch instead of sklearn**: in sklearn, the single line `model.fit()` hides the forward pass, loss, backpropagation, and update all inside it. KIM is the type who wants to see "the inside," so PyTorch — which unrolls those 4 steps and shows them — fits better.
- **Order of approach**: "reverse-engineer from the code" (X) → **"principles first in notes → confirm in PyTorch code → experiment by changing numbers"** (O). The same approach that worked well today.
- The starred (⭐) essentials: Day 7 (numpy backprop), 10 (CNN), 15 (attention), 22 (GPT), 25 (distributed training). The rest is the connective tissue between them.

## 57. The Skeleton of a PyTorch Training Loop (regression_pytorch.py)

- Day 2 (regression) converted to PyTorch. Same result as sklearn (MAE 43, R² 0.46), but the training loop is fully visible.
- **The core = the 4-line training loop.** The 4 steps from the hand calculation appear in the code exactly:
  ```python
  y_pred = model(X)              # (1) forward pass: input × weights → prediction
  loss = loss_fn(y_pred, y)      # (2) loss: mean of (prediction − answer)² = MSE
  loss.backward()               # (3) backpropagation: gradients computed automatically (that hand calc!)
  optimizer.step()              # (4) update: move in the gradient direction by the learning rate
  ```
- `nn.Linear(10, 1)` = **one layer** with 10 inputs → 1 output. Inside it's "multiply input × weights and sum it all + bias." The weights start random.
- `nn.MSELoss()` = the squared loss we computed by hand today. `optimizer` (SGD) = the part that moves the weights by the gradient, `lr` = the learning rate (the stride, set by a human).
- `optimizer.zero_grad()` resets the previous lap's gradients (without it, they accumulate). This is a PyTorch-specific gotcha.
- **The key realization**: sklearn's single line `model.fit()` was, in fact, the repetition of these 4 steps.
- This code is a single layer (linear regression), so there's no activation function or multiple layers yet. It's deliberately kept as simple as possible (to learn the loop skeleton).
- **Things to experiment with**: change `lr` to 0.01 (slower) / 2.0 (diverges?), increase the number of iterations to confirm convergence.

## 58. The Exact Difference Between Classification and Regression

- **The one-line difference**: whether the answer is a **number** (regression) or a **category** (classification).
- The core of linear regression is already familiar = "input × weight = prediction, a straight line." "Linear" = only multiplication and addition → a straight line (that thing we saw with activation functions). Statistical theory (least squares, etc.) is unnecessary for now on the deep-learning track.

| Aspect | Regression | Classification |
|---|---|---|
| Type of answer | A continuous number | A fixed category |
| Examples | House price, temperature, progress | Dog/cat, spam or not |
| Loss (grading) | (prediction − answer)², etc. — "distance" | cross-entropy — "did it give probability to the answer?" |
| Model output | A single number | The probability of each option (+ softmax) |
| Evaluation metrics | MAE, RMSE, R² | Accuracy |

- **The inner training loop is the same** (forward pass → loss → backpropagation → update). What changes is only how the loss is measured and the form of the output. (An instance of today's "the engine is the same; only input, answer, and architecture change.")

## 59. Regression Evaluation Metrics (MAE / RMSE / R²)

- All three are "how far off it was," but they **measure differently** → you usually look at them together.
  - **MAE**: on average, how far off it was. Stays in the original units, so it's easy to read. (for explaining to people / intuition)
  - **RMSE**: similar to MAE but, via squaring, **penalizes large mistakes more heavily.** (when large errors are critical)
  - **R²**: 0–1. "How much better than just guessing the average." 1 = perfect, 0 = meaningless. (a quick judgment of whether the model is usable)
- Priority: quick judgment → R², actual error → MAE, watching for large mistakes → RMSE.
- **You don't need to memorize the terms or formulas.** Just know "such things exist and roughly what they do," and look them up per project. Understand only the skeleton (the training loop, classification/regression, the meaning of loss) and fill in the details as you go.

---

## 60. How Sets, Layers, and Intermediate Values Relate (the easy-to-confuse core)

- **A weight set is not the same thing as a layer.** A set is "horizontal," a layer is "vertical."
  - **Weight set (horizontal)**: the teammates lined up side by side *inside* one layer. Each looks at a different feature. N sets → N intermediate values.
  - **Layer (vertical)**: the stages going measurements → intermediate values → final score. The previous layer's output is the next layer's input.
- **The rules**: number of sets = that layer's **number of outputs** / the size of one set = that layer's **number of inputs.**
  - E.g., layer 1 with 3 inputs (measurements) and 4 sets → 4 intermediate values. Layer 2 with 4 inputs (intermediate values) and 1 set → 1 final output.
- **Where does set C (layer 2's weights) come from?** They're layer 2's own weights, prepared when you build layer 2. They start random just like layer 1's sets. Their size is fitted to the number of incoming inputs.
- **One intermediate value = one feature** (a pointy ear, a long nose, etc. — an individual feature). **A layer = the stage that produces all of those features at once** (a layer is not one feature!).
  - Widening (more intermediate values) = looking at more features at the same stage.
  - Deepening (more layers) = combining features into a more complex level (line → curve → ear → face).

## 61. How Many Layers? / Only the Last Layer Is Fitted to the Output

- The number of sets and the number of layers are **hyperparameters a human sets in advance.** They don't change automatically during training (not dynamic). The counts are fixed; only the **values** inside the sets change through training.
- **Only the "last layer" is fitted to the number of outputs.** With 3 layers, layer 3 is fitted to the number of answers; with 4, layer 4 is. The middle layers are free to choose their number of sets.
  - E.g., 3-class classification → the last layer has 3 sets (a score per class). Middle layers can freely be 8, 4, etc.
- You stack layers deep (3, 4, or more) **the more complex the problem is.** Simple → 1–2 layers; complex → tens to hundreds. More isn't automatically better (it gets heavier and risks overfitting).

## 62. Layers Aren't "Competitors" — They're a Connected Pipeline / Inference Is Forward Pass Only

- **The layers don't each put forward an answer candidate and compete.** The intermediate values (5, 10, 18…) aren't answers compared against the label — they're **intermediate material passed to the next layer.**
- **The only thing compared against the answer is the single final output of the last layer.** "Which layer is closest to the answer" is meaningless (there's only one thing being compared, so there's no competition).
- Differentiation doesn't "pick a layer" — it adjusts **all the layers' weights together** so the final output gets closer to the answer. (A relay race: one final time, and you fix every runner's form together.)
- **Actual use (inference)**: you don't pick one layer — you **pass through all the layers in order** (layer 1 → 2 → … → final). This is the same as the forward pass during training, and it does **only the forward pass, no backpropagation** (which is why inference is lighter than training).
- When training ends, every layer's weights are frozen and saved = that's "the model" (70 billion parameters = the sum of all the layers' weights).

## 63. The Direction of Forward Pass vs. Backpropagation (layer 2's weights have nothing to do with layer 1's gradient)

- A common misconception: "layer 2's weights are layer 1's gradient multiplied in" → **no.** Each layer has its own weights and is fixed by its own gradient.
- A small example (measurement 2, w1 = 3, w2 = 4, answer 30):
  - **Forward pass (front → back)**: 2 × w1 → 6 (intermediate value), 6 × w2 → 24 (final). Layer 2 multiplies layer 1's **intermediate value 6** by w2 (not the gradient!).
  - **Loss**: 24 vs. 30 → difference −6.
  - **Backpropagation (back → front)**: layer 2's gradient first (using intermediate value 6) → then layer 1's gradient (which uses **w2's value as an ingredient**). Differentiation flows **layer 2 → layer 1.**
- In short: what gets multiplied is the **intermediate value** (forward pass); layer 2's weight being used on layer 1 happens in **backpropagation.** Two processes running in opposite directions.

## 64. How Do "Features" Arise from Random Weights? / Weight Initialization

- **Intermediate values aren't random** — they're the **result** of input × weight. What's random is the weights, and even those only **the very first time.**
- In the random state there are no features. **Features arise through training (the pressure to get answers right)**: "when this intermediate value responds to a pointy ear, it turns out to get more answers right" → differentiation adjusts the weights in that direction → that intermediate value becomes an "ear detector." Nobody ordered it; the pressure to get answers right shapes it.
- Why several intermediate values diverge into different features: if they all looked at the same thing, the information would be redundant — a loss → so they naturally differentiate.
- **Why more intermediate values (features) is good**: combining several clues gives more accuracy. Looking only at ears, you'd mistake a Shiba Inu for a cat, but adding nose and body size, you won't be fooled. But too many gets heavy and risks overfitting → "a moderate amount."
- **Weight initialization (a practical topic)**: if the sets are too similar — or **completely identical, which is the worst** — they all move the same way (the symmetry problem), and having multiple sets becomes pointless. So you **break the symmetry with randomness.**
  - But not just any randomness — **randomness scaled to a suitable size** (too large → unstable, too small → the signal dies out). Xavier / He initialization, etc. PyTorch's `nn.Linear` applies this automatically.

## 65. The Principle Behind Getting Closer to the Answer via Matrix Multiplication and Bending / History

- **A three-part collaboration**:
  1. **Matrix multiplication + bending (ReLU)** = preparing a "vessel (expressive power)" that can form any complex shape. (Multiplication = straight lines, bending = keeping the lines from collapsing together.)
  2. **Loss** = measuring how far the current shape is from the answer.
  3. **Differentiation → weight adjustment (training)** = reshaping that shape little by little to fit the answer.
- Analogy: multiplication + bending = a lump of clay (the material), training = the hands kneading it toward the answer's shape. You need both to get close to the answer.
- **Who invented it**: not one person's invention, but the sum of pieces across decades.
  - Most famous: **Rumelhart, Hinton, and Williams in 1986** applied and popularized backpropagation for neural networks (Hinton won the 2024 Nobel Prize in Physics → "the godfather of deep learning").
  - But the roots go back further: Werbos in 1974, and earlier math (the chain rule, control theory) before that. Who was first is still debated.
  - Why it didn't work before the 1980s: the belief that "neurons are 0/1" led people to use chopping (discontinuous) activation functions that couldn't be differentiated, plus there was distrust that gradient descent only finds local minima.

## 66. One-Hot Encoding — Categories as 0/1

- **The problem**: the model only handles numbers, but you need to feed in categories (text like BROAD/PHRASE/EXACT).
- **The bad way**: just numbering them (BROAD = 1, PHRASE = 2, EXACT = 3). → The model misreads the number's size as meaning ("EXACT is 3× BROAD," "PHRASE is in the middle"). It invents an order that isn't there.
- **One-hot encoding**: make as many slots as there are types, and set only the matching one to 1, the rest to 0.
  - BROAD → [1,0,0], PHRASE → [0,1,0], EXACT → [0,0,1]. Exactly one 1 per row (= hot).
- The effect: the types become **equal, independent switches.** No fake order or size. Each type's effect can be learned independently.
- **When to use it**: categories with few types (a handful to a few dozen) and no order (match type, channel, day of week, etc.).
- **Watch out**: too many types (hundreds) → the slots explode → use embeddings or another method. If there really is an order (small/medium/large), numbering may carry meaning.
- Analogy: a survey checkbox. Numbering blood types A/B/O/AB invites the "B is 2× A" misread → instead, check exactly one of four checkboxes.

## 67. When You Need Embeddings

- **When one-hot is enough**: few types + the items are on equal footing (no need to weigh similarity). E.g., match type, channel.
- **Two signals that you need embeddings**:
  1. **Too many types** → the one-hot slots explode (500 accounts → 500 slots, mostly 0). An embedding compresses them into a vector of just a few numbers.
  2. **"Semantic similarity" between items matters** → one-hot can't capture that "plumber and electrician are similar" (they're all unrelated independent switches). An embedding makes similar items into similar vectors, so you can learn "similar things → similar results." (The king − man + woman = queen principle.)
- **Important**: even when embeddings look necessary, **don't use them from the start.** First build a baseline with one-hot + hand-made features → add embeddings when performance falls short → compare. (Start simple; compare as you go.)
- Analogy: an organizer box. Few types → divide it into slots (one-hot); many types or "similarity" matters → use property coordinates (embeddings).

## 68. If You Change the Features (Inputs), You Have to Retrain the Model

- Adding an embedding, etc., means **the input features change = the number of inputs changes = the model architecture changes** (number of weights = inputs × outputs). → You can't reuse the old weights → **retrain from scratch.**
- This isn't a problem — it's the **normal, iterative improvement process**:
  - Tabular data retrains fast (minutes to tens of minutes). Possible several times a day.
  - The real-world flow = train a baseline → check the result → change features and retrain → compare → improve → retrain… repeated dozens of times.
  - To know "did this feature help," you have to train anew and **compare**, so retraining is essential.
- The baseline doesn't disappear — you keep it as a reference and compare the new version side by side (e.g., confirming R² 0.45 → 0.52).
- Analogy: improving a recipe. Add a new spice (feature) and you cook from scratch again to compare the taste. The base recipe stays as the reference.

## 69. What Do the Numbers Inside an Embedding Vector Mean?

- Each number = a **position (coordinate) along some "meaning axis."** Like a map's [latitude, longitude], except with tens to hundreds of axes.
- But those axes can't be named by humans (they're mystery axes the training created on its own = a black box). It's not the case that "axis 1 = industry."
- **You shouldn't try to interpret each individual number.** The real meaning lies in **how close the whole vector is to other vectors (distance/relationship).**
  - plumber [0.8,0.2,0.9] ↔ electrician [0.8,0.3,0.9]: close (similar) / dress [0.1,0.9,0.2]: far (different).
- Similar words are **scattered smoothly across a continuous space** (not split into sharp groups). That's what makes direction arithmetic like "king − man + woman = queen" possible.
- The value range isn't fixed to 0–1 (negatives and values above 1 appear too). Relative distance matters more than absolute magnitude.

## 70. Embedding Numbers Are Also Decided by Deep Learning / Embedding Dimension

- Embedding numbers = **learned weights.** Just like any other weight: random start → forward pass → loss → backpropagation → adjust → repeat.
- Why "similar words get similar vectors" = **the pressure to get answers right.** Treating "king" and "queen" similarly makes it predict the next word better → differentiation adjusts the two embeddings toward being similar. (The same principle as the dog/cat model coming to look at ears without being told to.)
- An "embedding model" is nothing special either — it's just a **deep-learning model whose purpose happens to be producing good embeddings.** The engine is the same.
- **Embedding dimension (vector length = how many numbers)**: a **hyperparameter a human sets in advance** (like the number of layers or sets). The values come from training; the count comes from a human.
  - Larger → more expressive power (capturing subtle meaning) but heavier and prone to overfitting. Smaller → lighter but underpowered.
  - Roughly: small-scale / categories 8–50, words 100–300, large models hundreds to thousands. **Start small and scale up, comparing as you go.**

## 71. Two Ways to Use an Embedding as Input (lookup vs. built-in)

- **Way A (lookup)**: pull the embeddings out in advance and use them as **fixed input features.** They don't change during training. Simple, reusable. → **Use this first for a baseline.**
- **Way B (built-in learning)**: put the embedding in as the prediction model's first layer and **train it together, toward the goal (e.g., CPC).** Optimized precisely for that problem, but more complex and needs a lot of data.
- Analogy: buying a finished translation dictionary and using it (A) vs. building your own dictionary for your problem (B). A first; B if A falls short.
- Note: even in way A, **adding** the embedding to the input **increases the number of inputs**, so the prediction model needs retraining (the embedding itself is fixed; the prediction model is retrained).

## 72. How to Feed an Embedding Array as Input / Matching the Scale

- **Even though it's an array, it just gets multiplied in.** The input [2,3,1] we learned about today was also an array, and each number in it was multiplied by a weight. An embedding [0.1,0.5,0.8] is the same — each number × each weight.
- **Combining several features = concatenating into one long array**: [word count, urgency] + embedding [0.1,0.5,0.8] → [word count, urgency, 0.1, 0.5, 0.8]. You unpack the embedding bundle and lay it out alongside the other features. No normalization is needed for the multiplication.
- **Embeddings aren't forced into 0–1** — you leave the values the model gave. Forcing them distorts the distances between vectors (the meaning).
- **What "matching the scale (normalization)" really means**: not "match it to the embedding," but **transform each feature independently by the same rule (e.g., mean 0, spread 1)** → so they all end up in a similar range. It's not cramming them into each other; it's measuring with the same ruler.
  - Method 1: leave the embedding alone and normalize only the other features (search volume, etc.). / Method 2: transform everything, embedding included, by one rule.
  - The goal: keep one feature from burying the others just because it's large (today's "the fake influence of a big number").
  - Analogy: to compare exam scores from different countries (out of 100 / GPA 4.0 / out of 20), you convert them all to the same basis — "percentile."

## 73. The Two Meanings of "Model" — Algorithm (Frame) vs. Finished Product

- The same word "model" refers to two things (hence the confusion):
  - **① The algorithm (frame)**: a **learning method / prediction structure** like "linear regression, decision tree, neural network." Not yet trained (the weights are empty). = a kind of recipe.
  - **② The trained result**: that frame with data fed in and **the weights fixed.** = the finished dish. (70 billion parameters is this.)
- **Linear regression, classification, trees = ① (frames).** You take a frame from a library and train it **on your own data** with `.fit()`.
  - `model = LinearRegression()` (the frame, an empty shell) → `model.fit(X, y)` (trained → finished product).
- **Transfer learning, off-the-shelf embeddings = ② (finished products).** You take something someone else already trained on massive data (ResNet, etc.).
- How to tell them apart: "before training / choosing an algorithm" → ① (frame), "after training / performance and results" → ② (finished product).
- Analogy: a car. ① = the model/blueprint ("shall we go with a sedan?"), ② = the finished car ("this car runs well"). Linear regression = take the blueprint and assemble it on your data; transfer learning = buy the finished car.

## 74. Each Frame Has Its Own "Learning Method" (there may be no weights at all)

- "Compute a gradient to get new weights" is **the method of the neural-network / linear-regression family only.** Not every frame works that way.
- What "learning" actually is differs completely by frame:
  - **Linear regression / neural networks**: adjust weights little by little via differentiation.
  - **Decision tree**: no weights! It finds "the question (if/else) that splits the data well." No differentiation either.
  - **KNN (nearest neighbors)**: barely trains at all. It memorizes the data, then finds similar examples and follows their answers.
  - **Gradient boosting (XGBoost, etc.)**: many trees, each new one correcting the previous one's mistakes. (Strong on tabular data.)
- The only thing in common is the goal ("predict well"); **the methods are all different.** So you try several frames suited to the problem and compare.
- That's why **layers, weight sets, and ReLU are all components of the neural-network family only** — trees don't have them (a tree has questions, branches, leaves). Today's deep dive into "why layers and ReLU are needed" is central to understanding neural networks but doesn't apply to trees.
- Analogy: reaching a destination (prediction) by car (neural network, steering adjustments), by asking at each fork (tree), or by asking someone experienced (KNN). Each vehicle has different parts.

## 75. Measuring Loss vs. Learning from Loss Are Separate / Loss Is Independent of the Frame

- **Loss is measured from "prediction vs. answer," not from weights** = `(prediction − answer)²`. Weights don't appear in the loss formula. So **a frame with no weights (a tree) can still have its loss measured, as long as there's a prediction.**
- The source of the confusion: thinking of "loss → differentiation → weight adjustment" as one bundle. That bundle is **the learning method of the neural-network family only.** A tree doesn't differentiate the loss to fix weights — it learns by "finding questions."
  - Separate **"measuring loss"** (prediction vs. answer, possible anywhere) from **"adjusting weights from loss"** (neural networks only).
- **Loss isn't set by the frame — you choose it separately, by the type of problem**: regression → MSE, classification → cross-entropy. Even for the same neural network, regression → MSE, classification → cross-entropy.
- Separating the three makes it clear: **① the frame** (prediction structure + learning method), **② the loss** (grading criterion, chosen by problem type), **③ learning** (the frame improving in its own way by looking at the loss). "Frame = loss algorithm" is wrong.
- Analogy: the student (the frame, the one solving problems) / the grading criteria (the loss, set by problem type) / the study method (learning, different for each student). The student ≠ the grading criteria.

---

## Hands-on Session: Forward + Backprop from Scratch in NumPy (2026-07-01, in progress)

> Code: [backprop.py](../code/backprop-numpy/backprop.py). Writing, line by line, a 2-layer neural network that solves XOR.
> Method: the AI does not implement it for me. **AI asks a question → I answer → if correct, that line gets added** — filling it in line by line.
> Why XOR: it can't be split by a single straight line, so it's the smallest problem that **requires a hidden layer + ReLU**. The goal is to prove "why stack layers and why bend with ReLU" in code.

**Questions I asked this session**

76. How do the derivative formulas in backprop work
77. What exactly is a hidden neuron — is it a weight set
78. Why does stacking 8 weight sets give `(8, 2)` (how to read a matrix's shape)
79. Why do we do matrix multiplication / does multiplying produce weights faster
80. What is bias and what does `b1` mean
81. Does the bias value go in randomly / what is `b1` at the first run

## 76. The Derivative Formulas in Backprop (for one layer)

- The core is the **chain rule**: stitch derivatives together from the back (loss) toward the front.
- Notation: `z = W·x + b`, `a = activation(z)`, loss `L`.
- It gets clean if you define `δ = derivative of loss L with respect to z`:
  - `δ = (dL/da) × activation'(z)`
  - Weight gradient: `dL/dW = product of input x and δ` (as matrices, `xᵀ · δ`)
  - Bias gradient: `dL/db = δ` (summed over the samples)
  - Signal to pass to the previous layer: `dL/dx = δ · Wᵀ`
- Activation derivatives: ReLU' is `(1 if z>0 else 0)`, sigmoid' is `a(1−a)`.
- Special case: with **sigmoid+BCE** or **softmax+cross-entropy**, the output-layer δ falls out to simply `prediction − answer`.

## 77. What a Hidden Neuron Is = One Weight Set

- One hidden neuron = **[one weight set] + [one bias] + [passing through an activation function]**.
- A neuron takes all the inputs, makes a weighted sum with its own set → adds bias → ReLU → emits **one number**.
- With 2 inputs, one set = 2 weights. 8 neurons = 8 sets = **8 columns** of `W1`.
- If all sets were identical, the 8 neurons would learn the same thing and be no better than 1 → so they start random and different (symmetry breaking).

## 78. How to Read a Matrix Shape `(rows, cols)` / Why Sets Stand as Columns

- `(rows, cols)` = (how many lines going down, how many numbers going across). It's just a label recording the "layout."
- Stack a set `[0.5, 0.1]` as **rows** 8 times → `(8, 2)`; stand them as **columns** and place 8 side by side → `(2, 8)`. Same sets, only the orientation differs.
- For `X @ W1` to work, W1's **rows = X's columns (2)**. So the answer is `(2, 8)`, with each set standing as a vertical **column**.
- Intuition: one cell of a matrix-mult result = the dot product of (one row of X: a sample's features) · (one column of W1: one neuron's set) = that neuron's weighted sum.

## 79. Why We Do Matrix Multiplication (it "uses" weights, it doesn't "find" them)

- Matrix multiplication doesn't produce weights. It's **the forward-pass computation that makes the weighted sum (prediction)**. Finding weights is done by backprop (differentiation) + the update (note 23).
- What it does = batching all the per-sample, per-neuron "multiply and add" into one operation (a double loop written as the single line `X @ W1`).
- The amount of computation (number of multiply-adds) is the same as the loop. It's faster because **hardware (BLAS, tensor cores) runs matrix multiplication ultra-fast and in parallel**. → not "weights come out faster" but "the prediction computation finishes faster."

## 80. What Bias Is / What `b1` Means

- Bias = the `+b` in `y = w·x + b`. Unlike the slope (w), it's the **starting height (y-intercept)**. A neuron's "default starting point before it looks at any input."
- Without it, the line must pass through the origin (0,0). In particular, when the input is `[0,0]`, the weighted sum is 0 regardless of the weights → without bias that neuron emits only 0 forever. With bias, `z = 0 + b = b`, so it can still respond.
- `b1` = the bundle of biases for layer 1 (hidden layer). One per neuron → 8. (`b2` is the output-layer bias; 1 output neuron → 1.)
- `np.zeros((1, 8))` = a zero-filled `(1 row, 8 cols)` array → `[[0,0,0,0,0,0,0,0]]`. `(1,8)` isn't "1 and 8 as two meanings" — it's **one shape (1 row × 8 cols)**. **8 (cols) = the 8 neurons (one bias each), 1 (row) = a single row for broadcasting** (bias is one set, independent of samples). `(8,)` (1D) would also work, but since z1 is 2D `(4,8)`, shaping bias as `(1,8)` keeps the alignment clean.

## 81. Why Bias Starts at 0 / What `b1` Is at the First Run

- `b1` at the first run = `np.zeros((1, 8))` = just **eight zeros** `[[0,0,0,0,0,0,0,0]]`.
- The sharp question: "You said bias is what keeps `[0,0]` from passing through 0 — but if it starts at 0, isn't z still 0 on the first step?" → Correct. **On the first step, `[0,0]`'s z1 really is 0.**
- The key shift: the reason bias exists is **not "to be nonzero from the start" but "to be able to become nonzero."** Bias is a learned parameter, so backprop pushes it from 0 to whatever value is needed.
- The decisive difference (why only bias can rescue `[0,0]`): for the `[0,0]` sample, the **weight gradient is 0** (`dW = input × δ`, and the input is 0). So weights can't learn anything from `[0,0]`. But the **bias gradient is not multiplied by the input** (`db = δ`). → Bias is the only parameter that can learn even from `[0,0]`.
- That's why it doesn't need to be random: starting at 0 is fine because backprop moves it. (Weights must be random to break symmetry, but bias has no such reason, so it's 0.) The initial value is just a starting point — both W1 and b1 get adjusted by training anyway.

## 82. Bias Is Not an Input — It's a Separate Parameter

- Numbers inside the model fall into two groups: **data** (input X, answer y — given from outside, changes per sample, not learned) vs **model parameters** (W1, b1, W2, b2 — owned by the model, shared across the batch, learned by backprop).
- Bias is a **parameter**. It isn't carried along with the input; it sits inside the model and is added during the forward pass. In `z1 = X @ W1 + b1`, only X flows in from outside; W1 and b1 stay put.
- Analogy (note 11): W and b = the fixed fixtures of a factory machine (tuned by training), X = raw material passing through. Bias is on the machine side.
- Why it's confusing = the **bias trick**: prepend a constant-1 fake input and treat bias as its weight, so `[1,A,B]·[b,w_A,w_B] = bias + weighted sum`. That makes bias look like it rides along as an input. But our code (and most modern frameworks) keep b separate.

## 83. Broadcasting — Why Arrays of Different Shapes Can Still Be Added

- Normally addition only works between **the same shapes**. But when shapes differ, NumPy automatically **stretches (copies) the smaller one** to match = broadcasting.
- The rule: compare the two shapes **from the right**. Each position is OK if it's ① equal or ② one side is 1, and the side that's 1 gets stretched.
- Example: `(2,3) + (1,3)` → the single row is copied across both rows. `[1,2,3] + 10 = [11,12,13]` is the same idea (a scalar stretched).
- **Where it's used in our code = the forward pass** `z1 = X @ W1 + b1`. `X @ W1` is `(4,8)`, `b1` is `(1,8)`. The rows don't match (4 vs 1), but the columns do (8 = 8) → b1's single row is copied onto all 4 samples = "the same bias for a neuron applied to every sample." Exactly the behavior we want.
- That's why b1 was shaped `(1,8)` rather than `(8,)`. And it doesn't actually copy memory — it just reuses the values → fast, too.

## 84. Forward-Pass Notation — z (weighted sum) and a (activation output)

- Each layer has **two steps**: **`z` = the weighted sum** (input × weights + bias, the raw value before activation), **`a` = z passed through the activation function** (the layer's final output).
- The trailing number is the layer index: `z1`/`a1` = layer 1, `z2` = layer 2.
- Our 2-layer flow: `X (4,2)` → `z1 = X@W1+b1 (4,8)` → `a1 = ReLU(z1) (4,8)` → `z2 = a1@W2+b2 (4,1)` → `output = sigmoid(z2) (4,1)`.
- Key: **`a1` (layer 1's output) is exactly layer 2's input.** Previous layer's output → next layer's input = what "stacking layers" means (notes 53–64). "z" and "a" are conventional names (z = pre-activation, a = activation).

## 85. Why We Need W2 When W1 Already Has 8 Sets (each layer has its own wiring)

- What W1's 8 sets produce is not "the answer" but **8 intermediate ingredients (a1)**. Combining those 8 back into "1 answer" also needs weights = W2.
- W1 = wiring from input layer (2) → hidden layer (8) (16 weights). W2 = wiring from hidden layer (8) → output layer (1) (8 weights). They connect **different pairs of layers**, so each has its own weights.
- Why collapse to 1: the final answer is 1 per sample (XOR 0/1), but after the hidden layer you get 8 numbers per sample. W2 weighs those 8 clues and combines them into 1 (note 74, "only the last layer is fitted to the output count").
- Analogy: the 8 hidden neurons = 8 observers each catching a different clue; W2 = the one judge combining them into a final call. (XOR can't be solved by a direct single layer — it needs the hidden layer + ReLU, then combining; notes 63–64.)

## 86. Why There's One Bias per Neuron (8) / Why It's Not a Fixed `+1`

- Misconception 1: bias = a fixed `+1`? → ❌ It's a **learned parameter** that starts at 0 and is adjusted by backprop (section 81). Not "+1" but "+b, with b learned."
- Misconception 2: one bias is enough? → ❌ There's **one per neuron**, so 8. Expanding z1, each neuron gets its own bias: `neuron j's z = (neuron j's weighted sum) + b1[j]`.
- If all 8 shared one `+1`, all 8 would shift identically and couldn't capture different features → the **same reasoning** as splitting weights into 8 sets (each must be independent).
- So b1 has as many as there are neurons = 8 → `(1,8)`. b2 has 1 (one output neuron) → `(1,1)` (here it really is a single value).
- Aside (shape visualization): W1 `(2,8)` holds each set as a **vertical column**, 8 of them (transpose = the `(8,2)` KIM pictured). W2 `(8,1)` is 8 values in a **vertical column** (each hidden neuron's contribution to the output).

## 87. b1 Has 8, b2 Has 1 — Bias Count = That Layer's Neuron Count / Where It's Added

- Rule: **each layer's bias count = that layer's neuron count = the number of columns in that layer's z.**
  - Hidden layer 8 neurons → b1 has 8 → `(1,8)`. Output layer 1 neuron → b2 has 1 → `(1,1)`.
- Where it's added: each bias is added at **its own layer's `+b` step** (right after that layer's matmul, before activation).
  - `z1 = X@W1 + b1` ← b1 at the hidden layer (the step that makes the intermediate values a1).
  - `z2 = a1@W2 + b2` ← b2 at the output layer (the step that makes the **final** value, not an intermediate one).
- Correction point: b2 is added at the final-output step, not an "intermediate value" step. The one added at the intermediate (a1) stage is b1.

## 88. Why the Output Layer Uses sigmoid, Not ReLU (activations differ by layer)

- Activation functions play different roles per layer. **Hidden layer = ReLU** (nonlinearity / bending, range 0~∞), **output layer = sigmoid** (turns the result into a 0~1 probability).
- XOR's answer is 0/1, so the final output must be a "probability of being 1 (0~1)" to be compared with the answer (loss). ReLU ranges 0~∞ so it could output `5, 100` — not a probability. sigmoid squashes any number into 0~1 (`sigmoid(0)=0.5`, `-∞→0`, `+∞→1`).
- So after the hidden layer, `a1 = ReLU(z1)`; after the output, `output = sigmoid(z2)`. (Correction: don't use ReLU at the output.)
- NumPy: `output = 1 / (1 + np.exp(-z2))` (`np.exp` = e to the power).
- `np.exp(x)` made simple: the special number `e(≈2.718)` multiplied x times (divided if negative). **Always positive, exp(0)=1, explodes for positive, approaches 0 for negative.** So the denominator `1+exp(-z2)` is always >1, keeping sigmoid strictly within 0~1 (big positive z2→1, 0→0.5, big negative→0).

## 89. What If There Are 3+ Layers / How Frameworks Build Layers

- **3+ layers**: the same pattern repeats. Each layer has a W·b connecting `(previous layer size) → (this layer size)`. E.g., sizes `2→8→6→1` give W1(2,8), W2(8,6), W3(6,1), with each b as `(1, layer size)`. The forward pass repeats `(matmul + bias + ReLU)`, with only the last using sigmoid.
- **Key rule**: adjacent layer sizes must interlock — the columns of the earlier W = the rows of the next W (earlier layer's output = next layer's input). That's what lets the matrix multiplications chain.
- **Frameworks**: you just "declare" layers, and the framework creates/initializes/trains the W·b automatically. One `nn.Linear(2,8)` = our `W1+b1`. You can list them with `nn.Sequential(...)` or build them in a loop. It also does backprop automatically via autograd.
- So doing it by hand is learning what happens inside the framework. (Detail: `nn.Linear` stores its weight as `(out,in)` and computes `x@W.T` — only the orientation differs.)

## 90. Who Decides How Many Hidden Neurons per Layer, and on What Basis (hyperparameters)

- **Who**: the human (engineer). It's not learned — it's a setting chosen before training = a **hyperparameter** (note 22). Weights are learned, but "how many neurons / how many layers" is set by a person beforehand.
- **Basis**: no fixed formula; mostly experience, convention, experiment.
  - Problem difficulty: complex patterns → more neurons/layers, simple → fewer.
  - Starting-point convention: often powers of 2 (8, 16, 32, 64…). A heuristic, not a law.
  - Adjust via validation: too few → underfitting (can't capture the pattern), too many → overfitting + slow + wasteful (notes 30, 31). Tune while watching the validation score.
  - Automated search: hyperparameter tuning (grid/random search, Bayesian), and further NAS (neural architecture search).
- Our XOR's 8: theoretically 2 hidden neurons suffice, but 8 gives margin for stable training. An example of the "enough + margin" heuristic.

## 91. z2 Is a Matrix (raw scores) / What the Intermediate Value Means / Turning Scores into Probabilities

- z2 isn't a scalar — it's a **`(4,1)` matrix**, one raw score per sample. exp and sigmoid apply **element-wise** (each cell separately). (The earlier "z2=5" was just one cell.)
- What a z2 value means = a **raw score (logit)**: how much that sample "leans toward class 1." Positive = toward 1, negative = toward 0, magnitude = confidence. But its range is unbounded, so it's **not yet a probability**.
- Turning score → probability is exactly what **sigmoid** does: it translates each score into 0~1. The sign decides above/below 0.5, the magnitude decides the distance. `z2=0` → exactly 0.5.
- Actual: z2 `[0, 0.525, -4.229, -1.842]` → output `[0.5, 0.628, 0.014, 0.137]`.

## 92. Why sigmoid Has the Shape `1/(1+exp(-z2))` / Why You Never Get 100% Confidence

- Note: what goes into exp is **`-z2`**, not z2. For z2=0.525, sigmoid uses **`exp(-0.525)=0.592`**, not `exp(0.525)=1.69`.
- Why "add 1 and divide 1 by it" = the device that **bounds the result to 0~1**. `exp(-z2)` is always positive (0~∞) → `1+exp(-z2)` is always ≥1 → `1/(1+...)` is always in 0~1. Big denominator → near 0; denominator near 1 → near 1.
- Example: z2=0.525 → `exp(-0.525)=0.592` → `1+0.592=1.592` → `1/1.592=0.628` (matches the actual output).
- Does z2=1 mean 100% confidence? ❌ `sigmoid(1)=1/(1+0.368)=0.73` (73%). To reach exactly 100% (=1) you'd need z2=+∞ → a finite score never lands exactly on 0 or 1, only approaches them.

## 93. Loss — MSE vs BCE for Classification, and Why BCE

- MSE = `mean((pred-answer)²)` — note 42's "square the difference and average." A valid loss.
- For classification (probabilities 0~1), the usual choice is **BCE (binary cross-entropy)** = `-mean(y·log(output) + (1-y)·log(1-output))`.
- Why BCE: ① punishes confident-wrong far harder (pred 0.99 but answer 0: MSE penalty 0.98 vs BCE 4.6). ② paired with sigmoid, the output-layer gradient falls out to just `pred - answer` (note 76). MSE multiplies in the sigmoid derivative → messier + vanishing gradient when saturated.
- So for this XOR (classification) we use BCE.
- Formula concept (no need to derive/memorize): `y·log(output)+(1-y)·log(1-output)` is a **switch** — if the answer is 1 only `log(output)` survives, if 0 only `log(1-output)` — picking "**the log of the probability assigned to the true answer**." The leading `-` flips it: high probability on the truth → penalty ≈ 0, low probability → penalty explodes. `mean` averages over samples. → "the lower the probability you gave the true answer, the bigger the penalty."

## 94. Is Backprop a Single Method or Many / What Does a Framework's `.backward()` Do?

- Backprop itself = **applying the chain rule back-to-front to get gradients** — one fixed algorithm. There aren't "many backprop methods."
- What varies is the **ingredients** it passes through: the loss (MSE/BCE — changes the starting gradient) and activations (ReLU/sigmoid — different derivatives). The resulting gradient formulas differ per network, but the **procedure (chain rule) is always the same**.
- Completely separate: the **update rule = optimizer** (SGD/Adam/RMSprop…) — that genuinely has many methods. Backprop = compute gradients; optimizer = take a step with them (notes 20, 35).
- PyTorch `loss.backward()` = **one general automatic-differentiation (autograd) engine**, not many methods — just the chain rule. Each operation carries its own derivative rule and is recorded in a graph → backward walks it back-to-front multiplying them → change the loss/activation and the same engine handles it automatically. The update is a separate `optimizer.step()`.
- Doing it by hand = doing what autograd does automatically.

## 95. What autograd Precisely Is (it differentiates the loss, it doesn't choose it)

- Misconception: autograd picks the loss based on the activation? ❌ The loss and activations are all **chosen by the human**. autograd chooses nothing.
- autograd = an **automatic differentiator**: it takes the computation you wrote (input → … → loss) and differentiates it to fill in gradients.
- How: ① during the forward pass it records each operation (matmul, +, ReLU, sigmoid, log, mean…) into a **computation graph** → ② each operation knows its own derivative rule → ③ `.backward()` walks the graph backward from loss to inputs, multiplying via the chain rule to fill in gradients. = **reverse-mode automatic differentiation**.
- Analogy: a calculator that, given any formula you type, computes its derivative automatically. You write the formula; the calculator differentiates it.

## 96. There Are Many Loss Functions / But They All "Measure, Then Differentiate"

- Beyond MSE·BCE there are many. By task: regression = MSE/MAE/Huber, binary classification = BCE, multi-class = Cross-Entropy, embeddings = Contrastive/Triplet.
- Common skeleton: ① measure how bad the prediction is (how much it disagrees with the answer) as a **single number** → ② differentiate that to move the weights in the reducing direction.
- Two requirements: **lower must be better** (so "reducing" = "doing better") + **differentiable** (so we can get gradients).
- Nuance: "distance from the answer" is the supervised case. Unsupervised/self-supervised has no explicit answer, so the loss measures a different criterion like reconstruction/clustering (note 32). And "distance" isn't always squared distance (BCE measures probabilistic "surprise"). But the spirit — "how far the prediction is from the truth" — is the same.

## 97. Do I Need to Understand the Algorithms and Math Proofs of Every Loss (binary/multi/embedding)?

- The depth depends on your **goal (role)**. Practitioner/engineer = **intuition** is enough (what each loss does, when to use it, why it behaves that way); proofs unnecessary (the framework computes them). Researcher = deeper math (deriving gradients, convergence, proofs).
- For KIM's current stage (fundamentals, intuition): "**understand the concept + do one by hand end-to-end**" is the answer. Finishing one (XOR backprop) opens the rest as "the same engine with a different loss." No need to re-derive each time.
- **Mathematical proofs/verification (convergence proofs, etc.) are not needed now** — that's research/theory territory (notes 4, 37).
- Karpathy's principle (README): don't sweep everything up front; learn on-demand when a project needs it. The math of multi-class (softmax+CE) or embeddings (contrastive/triplet) can wait until a project calls for it.
- Minimum: know the map (which problem → which loss) now; the deep math when the time comes.

## 98. Backprop Notation — the `d` Prefix in `dz2`, `dW2` / the Number Is the Layer

- `dz2` = **"the gradient of the loss L with respect to z2" (dL/dz2)**. The `d` prefix is a code convention meaning "gradient of the loss w.r.t. this." → Yes, dz2 is a gradient.
- What a gradient means (note 20): "if you nudge this value a little, in which direction and how much does the loss change."
- **The number 2 = the layer index** (same as z2·W2·b2 being layer 2). `dz2` = gradient at z2, `dz1` = gradient at z1. It's just the forward variable's name with `d` in front.
- Two kinds: **parameter gradients** (dW1, db1, dW2, db2 — actually used to update the weights) vs **intermediate gradients** (dz2, da1, dz1 — not parameters, just stepping stones passed backward through the chain). z2 isn't a parameter, so dz2 is an intermediate step toward dW2·db2.

## 99. The Principle of Derivatives/Gradients / Why output−y Is Simple / dz2 Is 4 Per-Sample Gradients

- **Gradient (derivative) = slope = "if you nudge the input a tiny bit, how much does the output change?"** Definition: `df/dx = lim(h→0) [f(x+h) − f(x)] / h` (a tiny "output change ÷ input change" = instantaneous slope). It's note 25's "nudge and see" computed exactly, without actually nudging. Example: the slope of `x²` at x=3 is 6.
- **For multiple steps, multiply (chain rule)**: `dL/dz2 = dL/doutput × doutput/dz2` — multiply each step's slope and stitch them together.
- **Why so simple (output−y)?** `dL/doutput` (BCE derivative, messy) × `doutput/dz2` (sigmoid derivative = output(1−output)); the messy parts **cancel algebraically**, leaving just `output − y`. Not a shortcut — the **simplified result of the full derivative** (sections 76·93). With a different loss/activation it wouldn't be simple.
- **dz2 (4,1) = 4 per-sample gradients** (not one). Each row = that sample's error signal. Parameter gradients dW2·db2 later combine all 4 samples into one gradient per weight (a weight is shared across all samples).
- Actual dz2 = `(output−y)/4` = `[0.125, −0.093, −0.246, 0.034]`. Sign `+` = predicted too high (lower it), `−` = too low (raise it); magnitude = how wrong. `[1,0]` is largest at −0.246 (most wrong).

## 100. Transpose `.T` — Swapping Rows and Columns / Why It's Used in dW2

- `.T` = **transpose** = flip rows↔columns. Element (i,j) → (j,i). Shape `(m,n)` → `(n,m)`.
- Example: `[[1,2,3],[4,5,6]]` (2,3) → `[[1,4],[2,5],[3,6]]` (3,2). Each row becomes a column.
- a1 `(4,8)` (rows=samples, cols=neurons) → `a1.T` `(8,4)` (rows=neurons, cols=samples).
- Why in dW2: ① shape — to make dW2 `(8,1)` you need `(8,4)@(4,1)` → transpose a1 to `(8,4)` (`a1@dz2` fails, inner 8≠4). ② meaning — dW2[i] (neuron i's gradient) = sum of "neuron i's outputs (column i of a1) · error dz2"; transposing turns column i into row i so the matmul's dot product computes it directly.

## 101. Why We Divide dz2 by 4 (the loss is a mean, so the gradient is ÷N too)

- The loss uses `np.mean` = the **average** of 4 sample-losses = `sum ÷ 4`. **Differentiating an average keeps that "÷4" in the gradient** → `dz2 = (output−y)/4`.
- Analogy: if `total = a+b+c+d` then `d(total)/da=1`, but if `avg=(a+b+c+d)/4` then `d(avg)/da=1/4`. That 1/4 just rides along.
- If we didn't divide? Gradients would be 4× larger = same as a 4× larger learning rate. It would still work, but wouldn't be the exact gradient of the mean loss we display.
- Why average (not sum) at all? Whether you have 4 or 400 samples, the gradient magnitude stays comparable, so **the same learning rate works** (not thrown off by batch size). That's why mean is standard.
- Picture: with 4 numbers a,b,c,d, nudging a by +1 → the sum rises +1 (slope 1), the average rises +0.25 (slope 1/4). The "+1 shared among 4" waters the effect down by 4 = ÷4. Drop a coin in one slot: the sum grows by 1, each person's share of the average by only 1/4.

## 102. What's in the loss Variable / When It's Used (a scalar · scoreboard)

- loss = **a single number (scalar)**. Current value = 1.3871. "How wrong the model is on average right now."
- When it's used: in our hand-written code, **only for monitoring** (print it to check "is it going down?"). The weight update uses the **gradients (dz2, dW2…)**, not the loss number.
- Key: we compute dz2 directly from output·y, not from the loss variable → in our code loss is effectively a **scoreboard**. (A framework's `loss.backward()` starts backprop from the loss node, but we derived the gradients by hand, bypassing the loss variable.)

## 103. Why loss Is 1 Number but dz2 Is 4 / When Samples Get Combined

- Confusion: "averaging should give 1 number" → yes, that's the **loss** (the average = one number, 1.3871). **dz2 is not the loss — it's the gradient**, so it's different.
- **loss (1)** = the average of 4 sample-losses → 4 collapse to 1. **dz2 (4)** = "how sensitive that one loss is to each sample's direction" → one question per sample → 4 answers. (Grades: final = 1 number; "improve each test by 1 → how much does the final move" = 4 answers, each 1/4.)
- **The 4 dz2 values are independent**: each row uses only its own sample's output·y. Sample 2 being very wrong doesn't change sample 0's value.
- **Samples get combined at the next step, the weight gradient (dW2)**: a weight is one knob shared by all 4 samples, so deciding how to change it requires gathering all 4 samples' errors. The matmul in `dW2 = a1.T @ dz2` sums across the 4 samples to make one gradient per weight. (Analogy: one volume knob, 4 listeners → gather all opinions into one adjustment.)

## 104. Is the Gradient "the Error Size" / Do We Multiply the Error onto the Weight to Move It

- Half right. ① The output error signal `dz2 ≈ prediction−answer` = the error itself (correct). But **a weight's gradient dW = error × the input that weight multiplied** (a1). Not "error alone" but "error × input," so even a big error gives zero gradient if the input was 0 (a dead neuron). = "how much this weight contributed to the error (its share of blame)."
- ② Moving is subtraction, not multiplication: `new weight = old − learning_rate × gradient` (`W = W − lr·dW`). We don't multiply the gradient onto the weight; we take one step in the opposite direction of the gradient (toward lower loss), sized by the learning rate.
- Why `−`? The gradient points toward higher loss → to reduce it, go the opposite way (note 35). Why small (lr)? Too big a step overshoots and diverges → small steps, many times (notes 35, 46).

## 105. The Training Loop + Result — XOR Actually Solved

- ⑥ update (4 lines: `W1 -= lr*dW1` etc.) + ⑦ wrap the whole thing (forward→loss→backprop→update) in a `for` loop for **10000 iterations**.
- Structure: data & weight init **once, outside**; predict→loss→backprop→update **repeated inside the loop**; print loss every 1000.
- Result: loss **1.3871 → 0.0001** (near zero). Final predictions `[0,1,1,0]` = answers `[0,1,1,0]` **exactly**. (The [1,0] sample, first predicted 0.014 — the opposite — is corrected to ~1.)
- Meaning: proves the hand-written forward + backprop **actually works**. Solves XOR, which a single layer can't, thanks to the **hidden layer + ReLU** (notes 63, 64 "why stack layers + why ReLU" proven in code).
- Setup: lr=1.0, 10000 epochs, seed 42. By epoch 1000 the loss was already 0.0015 — essentially solved.

## 106. What Exactly Is the Gradient Number / lr=1 Doesn't Mean "Move by 1," It's "gradient × lr"

- A gradient is not a position but a **rate of change (sensitivity)**: "if I nudge this weight up a tiny bit, how much does the loss change?" E.g. a dW2 value of −0.34 = "raise it by 1 → loss changes ~0.34," and the − sign = "raising it lowers the loss."
- Sign = direction: **+** = raising it raises loss (should lower it) / **−** = raising it lowers loss (should raise it). So we always move **opposite** to the gradient (the minus in `W -= …`, note 35).
- "Do we compute the level of deviation from 0?" Half right. The real deviation starts at the output's `output−y` (the error). A gradient of 0 means "flat = at the bottom," so no move. There's no separate step that compares the gradient itself against 0.
- **Step = lr × gradient**. Even with lr=1 you move by "the gradient," not by 1. Each weight moves a **different** amount, by its own gradient (0 gradient → no move). lr is just the shared step-size multiplier on all of them (notes 35, 46).

## 107. The Relationship Between the Weighted Sum z2 and the Answer y — Not Directly Comparable, sigmoid Is the Bridge

- z2 (the weighted sum) is a raw score from −∞ to +∞; y is 0/1. **Different languages (ranges), so you can't subtract directly.**
- The bridge = **sigmoid**: it squishes z2 into 0~1 (output = probability of being 1), translating it → only then is it in the same language as y and comparable.
- Flow: `z2 → sigmoid → output → (compare) → y`. z2 and y have no direct relation; they must meet through output. The error (output−y) comes after (notes 76, 99).

## 108. How sigmoid Works / Why z2=0 Gives 0.5 / 0.5 Is Not a Derivative

- sigmoid = a **squishing (transformation) function**, not differentiation. It packs −∞~+∞ into 0~1.
- The key is the denominator's `e^(−z2)`: e^(anything) is **always positive** → the result never leaves 0~1. Big positive z2 → e^−z2≈0 → ~1; big negative → e^−z2 explodes → ~0.
- z2=0 → `1/(1+e^0)=1/2=0.5`. Meaning: the weighted sum is balanced = "50/50 between 1 and 0, don't know." The exact center of the S-curve.
- Why e? ① the result can't escape 0~1 ② it's smooth so **differentiation later (backprop) comes out clean** (the real derivative shows up in dz2, note 99).

## 109. output Is "the Probability of Being 1" — Low Isn't Bad / The Criterion for "Well Matched"

- Read output in **one direction only: "probability of being 1."** 0.14 = "14% chance of 1" = "86% chance of 0" (confident it's 0).
- **Well matched = output close to y** (not high/low per se). If y=1, higher is better; if y=0, lower is better. E.g. output 0.14 & y=0 → well matched. output 0.01 & y=1 → confidently wrong the other way (biggest penalty).
- Distinction: **output = probability** (a prediction made without looking at the answer), **output−y = error** (the result of comparing with the answer). Two different things.

## 110. When the Answer y Isn't 0/1 — Swap the Last Layer + Loss by Problem Type

- Principle: **the last transform (sigmoid's slot) and the loss are chosen as a pair to match "y's shape."** The skeleton (forward→loss→backprop→update) stays the same; only these two are swapped.
- y a **real number** (house price, temperature) → regression: **z2 as-is, no sigmoid** + **MSE** (squishing would make it impossible to represent a value like 500M).
- y **one of several classes** (dog/cat/bird) → **softmax** (a probability distribution summing to 1) + **cross-entropy**, with y one-hot `[1,0,0]`.
- The current sigmoid+BCE is just "the case where y is 0/1" (note 96). sigmoid+BCE also works if y is a real number in 0~1 (soft labels).

## 111. loss Sits at the End of the Forward Pass (Just Before Backprop) / dz2 Is Its Derivative — Same Root, Different Role

- Computing loss is **not backprop but the final knot of the forward pass**. Backprop **starts from** that loss. Order: predict→[loss]→differentiate (dz2)→…. loss sits right at the boundary before backprop.
- Regression is the same: MSE at the end of the forward pass, `dz2 = 2(z2−y)/n`. In both binary and regression, **the starting signal ends up being "prediction−answer"** (sigmoid+BCE and linear+MSE are pairs whose derivatives line up cleanly, notes 93, 99).
- `dz2 = (output−y)/n` is **the derivative of BCE with respect to z2** → same root as loss (the intuition is right). But **loss = one scalar for watching**, **dz2 = the differentiated gradient (the actual worker)** — different roles (note 102). The loss line (:39) is only used for printing; delete it and training still works — dz2 comes straight from output·y.

## 112. Using All 4 output−y Values for the Gradient — Sign Gives Direction, One Weight Is Shared by 4 Samples So They're Combined

- Actually, **the sign already gives each sample's direction**: + = predicted too high (push toward 0, ↓) / − = too low (push toward 1, ↑). Even one value like 0.5 tells you the direction.
- Still, **why use all 4**: one weight **affects all 4 samples** → to decide "which way for this one," you must gather all 4 samples' demands (which **can conflict**) to get the net direction.
- Where they combine = the matmul in `dW2 = a1.T @ dz2`: `dW2[j] = Σ a1[i,j]×dz2[i]` (sums the 4 samples, weighted by how active the neuron was for each, notes 100, 103).

## 113. The Fundamental Reason for Dividing dz2 by 4 — Sum vs Average / "÷4 Isn't Mandatory (lr Absorbs It)"

- ÷4 = turning a sum into an **average** (note 101). Experiment: duplicate the XOR data to 8 samples → the **sum** goes −0.72→−1.44 (2× → the step doubles → same data, yet it lurches) / the **average** stays −0.72/4 = −1.44/8 = **−0.18**. → the step isn't swayed by sample count.
- **Key (new conclusion this session): ÷4 is convenience, not correctness.** Using `output−y` as-is still trains (the intuition is right). ÷4 keeps the **direction the same and only shrinks the magnitude to 1/4**.
- That magnitude is **absorbable by lr**: ÷4 = making lr 4× smaller. Proof (db2): `1.0×(−0.18) = 0.25×(−0.72) = −0.18` (same result). The two knobs (÷N, lr) are really one.
- The benefit of ÷N: **it decouples lr from the number of samples** → the same lr works regardless of batch size (without it you'd re-tune lr every time the count changes). That's why averaging is standard.
- Rounding caveat: the comment numbers are **rounded to 2 decimals**. `0.50/4=0.125` shows as `0.12` in the comment. The first sample `[0,0]` has zero input and zero bias, so z2 is **exactly 0** → output **exactly 0.50** → dz2=0.125 (shown as 0.12).

## 114. lr Is a Step Multiplier (gradient×lr) / The Update Is Subtraction, Not Multiplication / Overshooting

- lr is not the step itself but the **multiplier on the gradient**. Even with lr=1 you move by "the gradient," not by 1; each weight moves by its own gradient (note 106).
- **Update = subtraction**: `new W = old W − (lr × gradient)`. Not weight×gradient. Each weight is independent, so summing the moves into one "total" (e.g. 6) is meaningless.
- **÷4 (or lowering lr) keeps direction & ratio, shrinks only the size.** Not "more balanced" but "all smaller." The move ratios (e.g. 1:2:3:4) stay identical; only the overall scale is 1/4.
- **Overshooting**: too big a step leaps over the loss-valley floor and lands higher on the far side → bounces back and forth and diverges (loss blows up / NaN). Dropping ÷4 makes the effective lr 4× bigger (risky); shrinking lr to 0.25 restores safety (note 113).

## 115. dW2 = a1.T @ dz2 — Gradient = Input × Error / dW2 Is Not a "New Weight" but an "Instruction"

- **Gradient = (the input that flowed through the weight) × (the error out of it).** Forward z2=a1@W2, so W2[j] multiplied a1[j] → ∂z2/∂W2[j]=a1[j] → chain rule `dW2[j] = dz2 × a1[j]`.
- **Why a product?** Both input and error must be nonzero to contribute; if either is 0 the gradient is 0 (a ReLU-dead neuron a1=0 → zero gradient even with big error).
- Measured dW2[0] = sum over samples of (a1×dz2): only sample 2, active (0.50) × big error (−0.246) = −0.12, dominates; dead samples 0·1 contribute 0. The matmul does 8 weights × 4 samples at once (notes 100, 112).
- ⚠️ **dW2 is not a new weight.** Backprop (42–48) only computes each weight's **gradient (d__) = an instruction**. New weights are made in the **update step** (`W2 -= lr*dW2`, 51–54). Flow: dz2(error)→dW2(gradient)→[update]new W.

## 116. Why "Contribution" Is Called "Gradient" (contribution = sensitivity = slope)

- A gradient literally means "how much the loss changes if you nudge this value" = (Δloss)/(Δweight) = slope (notes 99, 106).
- **Contribution = gradient = the same number**: a big contributor = nudging it swings the loss a lot = steep slope; zero contribution = nudging does nothing = flat (zero gradient).
- Flow: nudge W → z changes by (input) → loss changes by (input×error). That "input×error" (the contribution) IS "how much the loss changes when nudged" (the gradient).
- Picture: x-axis = W value, y-axis = loss curve. dW = the slope of that curve at the current point — hence "gradient." Being a slope, it tells the direction & size to descend, used in the update.

## 117. db2 = np.sum(dz2) — Why the Bias Gradient Is Just a Sum / The Purpose of Bias

- **The bias gradient is simple because its "input" is 1.** Forward z2=a1@W2+b2, b2 is just added (not multiplied) = ×1 → ∂z2/∂b2=1 → `db2 = 1×dz2 = dz2` → sum over 4 samples = np.sum(dz2). (Number: sum = −0.18.)
- **Same formula as dW2** (Σ input×error); for bias the input is 1 so the multiply disappears, leaving a plain sum. Not "we chose to keep it simple" — the math just came out simple.
- **Purpose of bias = intercept (offset)**: removes the forced pass-through-origin (y=ax → y=ax+b); z can be nonzero even when input is 0. Shifts the ReLU firing threshold.
- The purpose (intercept) and the reason the gradient is simple (input=1) are **separate**.

## 118. da1 = dz2 @ W2.T — Returning the Error to the Hidden Layer (symmetry rule, intermediate gradient)

- **da1 = weight (W2) × error (dz2).** Forward z2=a1@W2, so ∂z2/∂a1[j]=W2[j] → `da1[j] = dz2 × W2[j]`.
- **Symmetry rule**: reversing a multiply = **error × the OTHER input**. dW2 multiplies by its partner a1; da1 multiplies by its partner W2.
- ⚠️ **da1 is neither a new weight nor for updating** → an **intermediate gradient** (note 98): the new "error signal" returned to the hidden layer, a stepping stone to reach W1 (the hidden layer takes over dz2's role). a1 is not a parameter.
- **W2.T (transpose)**: flips the forward direction (8 neurons→1 output) into the backward direction (1 output error→8 neuron errors). Shape (4,1)@(1,8)=(4,8) = same as a1. Each row = dz2[sample]×W2 (one output error spread over 8 neurons by W2's size). Measured da1 values added to the code snapshot.

## 119. What Exactly Is a Hidden Layer

- A neural net = **input layer → hidden layer → output layer**. The hidden layer = the **intermediate computed layer** between input and output. In this code, **a1 (8 neurons)**.
- **Why "hidden"**: you feed the input and read the output, but the middle a1 is computed internally — you neither see nor set it. The network decides it on its own during training.
- **What it does**: builds intermediate features between input and output (pixels→[ears·nose·fur]→dog/cat). Solves XOR (unsolvable by a single layer) via hidden layer + ReLU (notes 63, 64, 105).

## 120. What the Weighted Sum (z) Means — Not a Probability or an Error but a "Combined Score"

- z = a **combined score (evidence)**: it weights each input by its importance and sums into one number = "how much of this neuron's feature is present." Analogy: a weighted grade average (mid×0.3+final×0.5+hw×0.2).
- **Not a probability** (that comes after sigmoid, 0~1). **Not an error** (error = output−y needs the answer y; z is computed without y).
- **"Contribution" view**: each term (input×weight) = one contribution; z = the **sum** of contributions. z is the "sum," not a "contribution" itself.
- ⚠️ Distinguish two "contributions": **forward contribution** (input→score, this z) vs **backprop contribution** (weight→error = gradient, notes 115, 116).

## 121. Why Backprop Works "Exactly" — One Connected Function + Gradient Checking

- **Principle**: the whole network is one connected function (weight→weighted sum→ReLU→…→prediction→loss). Nudging any weight sends a ripple all the way to the loss, exactly.
- **Why exact?** Each step's local slope is exactly known (multiply→input, ReLU→0/1, sigmoid→out(1−out)). Multiplying them along the path (chain rule) = the exact weight→loss effect. Not an approximation.
- **Proof = gradient checking (poke it)**: nudge a weight by ε and measure `(L(w+ε)−L(w−ε))/2ε` — matches the backprop gradient to 12 decimals (verified). This is how you check a hand-written backprop.
- **Why the weighted sum is central**: z=w·x, so ∂z/∂w=x → the forward ingredients (inputs X·a1) are exactly the backprop gradient's ingredients. The weighted-sum structure is what makes the gradient computable (dW2=a1.T@dz2, dW1=X.T@dz1 reuse the inputs).
- A zero gradient is also "exactly zero": nudging a weight feeding a dead neuron doesn't move the loss (measured 0 too).

## 122. Single Layer (No Hidden Layer) = Logistic Regression / The "1" in "Probability of 1" Is the Label

- With no hidden layer this layer is the output layer → weighted sum → sigmoid → prediction. = **single layer = logistic regression**. E.g. sigmoid(−0.47)≈0.38.
- Activation depends on the **layer's role**: hidden = ReLU (pass/block features), output = sigmoid (0~1 probability, to compare with y). sigmoid is normally only for the last layer.
- **The "1" in "probability of 1" = one of the two label values (0/1) of the answer y** (not a probability or a count). sigmoid output = P(answer=1); P(0)=1−output. In XOR, "1" = the two inputs differ. (0/1 are just names for the two classes, as numbers so the math works.)

## 123. How a Single Multiplication Classifies 0/1 — Learned Weights = the Rule, Weighted Sum = a Straight Boundary

- **The current prediction (0.38) is meaningless**: the weights are random (seed 42). For predictions to be right the weights must be **trained** (= why backprop exists). The multiplication doesn't "know"; the **trained weights store the rule** and the multiplication just applies it.
- **Mechanism**: the weighted sum = an "evidence score for 1." Positive weight = a bigger input pushes toward 1, negative = toward 0. z big positive → sigmoid ~1, big negative → ~0, 0 → 0.5.
- **Training = adjust weights so z comes out positive for 1-samples and negative for 0-samples.** One weighted sum = one straight boundary; training = drawing that line well.
- ⚠️ **XOR is not linearly separable** (no single straight line separates ●/○) → **impossible for a single layer** → needs a hidden layer (notes 63, 64, 105, 119).

## 124. The Geometry of the Decision Boundary — the Line Is z=0, the Slope Is −w1/w2 (not the weights themselves)

- **One weight set = one line (decision boundary).** Random weights = a line at a random angle & position.
- **The line itself = where z=0.** z is not the line's endpoints but **which side of the line each point is on, and how far** (a signed distance). On the line=0, one side=+ (the "1" side), the other=− (the "0" side); farther → bigger |z|.
- Line equation: `w1·x1+w2·x2+b=0` → `x2 = −(w1/w2)·x1 − b/w2`. **Slope = −w1/w2 (the ratio of the two weights, not a weight itself); intercept = −b/w2 (set by bias).** Neuron 0 (0.50,−0.47,b=0) → slope 1.06, through the origin.
- **Coordinate meaning**: x-axis = x1, y-axis = x2 (the two input features). **z is not an axis** but a value computed at each point (a "height"). The line is drawn from any two points ((0,0),(1,1.06)).
- The weight vector (w1,w2) is **perpendicular** to the line, pointing toward increasing z (the "1" side).

## 125. AND, but the Line Sometimes Fails to Find the Answer — Not a Capacity Problem, an Optimization One

- **Key reframe**: unlike XOR, AND is **linearly separable** (one line suffices, note 124). So a correct line **exists** → failing to find it isn't about capacity, it's that **gradient descent doesn't reach that line**.
- **What "sometimes" means = random initialization**. Depending on the starting weights it rolls into a good valley or gets stuck on a flat plateau. That's why resetting changes the outcome.
- **Three forces that trap it on the plateau**:
  1. **3:1 imbalance** — AND has three 0s and one 1. Just predicting "all 0" already scores 3/4 (75%). A low-loss trap.
  2. **Lazy solution (horizontal line)** — the screen's horizontal boundary uses only x2, ignoring x1. Treats [1,1] and [0,1] (both above) as 1 → only [0,1] is wrong (3/4). The correct AND line is **diagonal** (cuts off only the [1,1] corner).
  3. **Sigmoid saturation → vanishing gradient** — once the 3 points are confidently classified, their gradient ≈ 0, so there's almost no force left to rotate the line → the loss curve drops early then goes **flat** (the curve at the bottom of the screen is the evidence).
- lr=1.00 is also on the high side → can bounce around near the imbalanced point.
- **Leads (to try next)**: reset to re-initialize, lower lr, run more steps, add hidden neurons.

## 126. The Drawn Line Gives a "Probability" First; "Correct or Not" Is the Result of Cutting at 0.5

- It's both, but in **order**: **probability first**, then "correct or not" is one more step on top.
- New input → signed distance z from the line (note 124) → sigmoid(z) = **P(label=1), a 0–1 probability** (note 122). This is what the network **directly** outputs → "probability of being correct given the line" is the real output. (The net doesn't say yes/no; it says "1 with probability 0.87".)
- The final 0/1 decision = put a **threshold of 0.5** on the probability: ≥0.5 → 1, <0.5 → 0. **0.5 = z=0 = the line itself** → "which side of the line" is exactly the yes/no.
- The screen's "3/4 correct" is also computed by cutting the probability at 0.5 and comparing to the true label.
- Summary: **the line = a probability map** (continuous). The hard answer is a 0.5 partition placed on top of it.
- (KIM's own derivation) "**decide by the nearer side**" = the 0.5 threshold. 0.8→1, 0.2→0. Exactly 0.5 (on the line) is ambiguous → by convention ≥0.5 counts as 1.

## 127. With Many Neurons (Lines), Which One Gives the Probability? — Not Any Single Line; the Output Neuron Combines Them All

- N hidden neurons = N lines. But **the final probability isn't read off any single line**. There is still just one final probability.
- Structure: input → **hidden layer** (several neurons, each one line = a partial judgment "which side of my line is this point?" → one activation value, e.g. ReLU) → **the output neuron takes another weighted sum of those values → one final z → sigmoid → one probability**.
- Hidden lines = intermediate judges (partial questions), not the final answer. **The output neuron synthesizes** them into a single probability.
- On screen: **colored lines = each hidden neuron's boundary** (intermediate) / **shaded probability field = the final output** (synthesis). The field's boundary can be **bent or curved** rather than straight because it's several straight lines combined.
- This is how XOR gets solved: what one line can't separate, several lines combined carve up with a piecewise/curved boundary (notes 63, 64, 105, 119).

## 128. The Probability Field = a "Map" Coloring the Whole Plane — Any Input Gets an Answer, but "Correct" Is Only Defined at the Trained Points

- The field pre-colors the **entire input plane**, not just the 4 points. Any point like (0.3,0.9) gets an answer just by "which colored region am I in?" → no per-input search, just **plugging into a fixed boundary** (= inference).
- Two meanings of "reach the answer":
  - (a) Does it **produce an answer** for any input? → Yes, always (every point has a probability).
  - (b) Is that answer **always correct**? → A different matter.
- AND's "correct answer" is only defined at the 4 corners (00·01·10·11). (0.5,0.5) has **no** defined answer → what the net outputs there is its learned rule **extended (a guess)** = **generalization**.
- **Infinitely many** lines separate the 4 points equally well → all correct on the 4 points, but they disagree in the empty space between. So "always correct for any input" is not guaranteed.
- Precondition: if training fails in the first place (note 125's 3/4 horizontal line), it won't even get the 4 points right.

## 129. Why Training Stops (the Line Stops Moving) at Some Point — Because the Gradient Approaches 0

- Amount the line moves per step = **lr × gradient**. lr is fixed → **when the gradient nears 0, the line stops**. (KIM's derivation)
- gradient = input × 2 × (prediction − answer) [regression form]. Of the factors that can zero the product:
  - **(prediction − answer) = 0 → healthy stop (convergence)**: prediction = answer = nothing left to fix = loss at minimum = training complete. ← the main reason.
  - **input = 0** → a real effect (note 121, dead neuron/input) but **local** (only that weight for that sample). Other samples & the bias (input always 1) keep moving → not why the whole net stops. (KIM spotted this.)
- **The other kind of stop (stuck)**: stopping even though not everything is correct yet = **vanishing gradient / saturation**. In a multi-layer net, when a hidden neuron saturates (sigmoid at its extremes) or dies (ReLU input negative→0), the gradient along that path ≈ 0 → the line won't move even with error remaining → plateau. The AND 3/4 stall is this case (note 125).
- Two kinds of stop: **A. all-correct** (prediction−answer=0, loss≈0, convergence) vs **B. stuck** (vanishing gradient, loss stays high, plateau). Tell them apart by whether the loss curve's drop→flat lands near 0.

## 130. Why Too Many Neurons Is Bad — Cost + Redundancy + Overfitting

- (KIM already knew) **Cost**: more multiplications, memory, time.
- **Redundancy/waste**: many extras just do the same job. On the XOR 4-neuron screen, neurons 1·2·3 have nearly identical lines (slopes 1.07/1.03/1.06, intercept≈0) → only 2 truly distinct lines. No added power, just waste.
- **Overfitting** ← the key: more neurons = more ability to draw a very wiggly boundary → fits the training points perfectly but **memorizes points instead of learning the rule** → wrong on unseen points (note 128, generalization). Analogy: a student who memorizes answers verbatim (100 on seen problems, 0 on slightly changed ones). Especially dangerous with lots of noisy data.
- **"If it's correct even by memorizing, isn't that good?" → No** (KIM's derivation: the goal is **prediction**). Training points are ones we already know → a table would do (not prediction). Memorizing scores 100 on training but **fails at the actual goal: new inputs**. Same 100 differs inside: **learned-the-rule 100** (new inputs right = generalization, smooth boundary) vs **memorized 100** (new inputs wrong = overfitting, wiggly boundary). How to detect: test on held-out data (test set) → high on training but low on new = overfitting.
- So a **moderate number** is best: too few can't solve it (underfitting, note 125's single line), too many overfits/wastes. XOR needs at least 2 (one line can't do it, note 127).

## 131. Why Memorizing Makes Prediction Wrong — Training Pins the Boundary "Only at the Points," but New Inputs Land "Between the Points"

- KIM's pushback: "the lines are near the answer, so why can't it predict?" → the trap: "near" holds **only at the training points**, not in the space between them.
- Two facts: ① new inputs rarely land exactly on a training point → mostly in the **empty space between** ([0.9,0.85] etc.). ② training (loss) pins the boundary **only at the training points** → it doesn't care what the boundary does in the empty space (no answer there, note 128).
- → **neuron count = how wildly the boundary can wiggle in the empty space**. Few = simple (smooth) → empty space stays tame → new points sensible. Many = to pass exactly through training points it bulges/spikes/wiggles between → a new point there gets caught and misclassified.
- **Connect-the-dots analogy**: 7 dots. Rule = a smooth line (in-between is natural) vs memorize = a curve passing exactly through all 7 but thrashing up/down between (7 perfect, in-between garbage). Both "pass near the dots" but **differ completely between them**. New inputs live between.
- Plus noise: if a training point is mislabeled, many neurons contort the boundary to enclose that wrong point → a nearby correct new point gets caught and misclassified.
- Correction: prediction isn't "impossible" (the map always outputs, note 128) — the output is just **wrong**. Not "can't," but "can't be trusted."

## 132. In Practice: What Dataset & Labels an Image Object-Recognition Model Needs

- First, **defining the task** determines the label shape: image classification (one class per image) / multi-label (several classes) / **object detection** (class + box coordinates per object, "what and where"). "What objects are present + where" = usually detection.
- Needed: thousands–millions of images + a **correct label** per image (note 128: an answer must be defined to train/score). Labels are added **by humans** (annotation) — for detection, a box + name per object. The most expensive part.
- **Data diversity is key** (angle, lighting, background, kind): narrow data widens the "empty space" → wrong on new photos → diverse = filling the empty space = generalization (note 131).
- **train/val/test split**: realizes "test on held-out data" (note 130). Train / validate / final test.
- Famous examples: ImageNet (classification), COCO (detection).

## 133. How to Decide the Number of Neurons (Layers/Size) — No Formula; Experiment via a Validation Set

- **No fixed formula. Found by experiment.** Compass = **validation score** (train score alone can't reveal overfitting, note 130).
- Diagnosis: train↓·val↓ → underfit → **increase**. train↑·val↓ → overfit → **decrease or regularize**. train↑·val↑ → just right.
- In practice: not from scratch — **start from a proven existing architecture (ResNet etc.)** → adjust while watching val.
- Lots of complex data → bigger / little data → smaller (avoid overfitting). Goldilocks (note 130).

## 134. Input Is Normalized, but Turning the Output into Text Isn't "Un-normalizing" (Classification) — It's a Label Lookup

- **Input normalization ✓**: pixels 0–255 → 0–1 (or mean0/std1). Prevents large values from wrecking training; numerically stable.
- **The output (classification) is NOT a "normalized answer value"**: it's **one probability per class** (softmax/sigmoid, note 126 extended to many classes; multiple outputs = note 127). E.g. [cat 0.8, dog 0.15, bird 0.05].
- **Text conversion = argmax (the slot with the biggest probability) → look up that slot's label** (cat=0, dog=1… table that we defined). **Not an inverse-normalization** — the text was never normalized into a continuous number; it's just a slot index/label (note 122: "the 1 is a name tag, not a count/probability").
- **Exception = regression**: if the output is a continuous value (house price, temperature, detection box coords) and targets were normalized → the output is normalized too → **inverse-transform it back** (here KIM's intuition is right). Object detection uses both: name (classification→lookup) + box coords (regression→inverse-transform).

## 135. If Only Colors (Pixels) Go In, How Do "cat/dog/bird Probabilities" Come Out — The Output Slot's Meaning Is a Name Tag We Assign

- Input is only colors (numbers); the word "cat" never enters the net. The 3 outputs are just **3 numbers**; the net doesn't know "cat."
- **"slot 0 = cat" is a name tag we chose** (note 122). Same as XOR: output "1" = the two inputs differ, but the net didn't know the word — it just learned to output high when they differ.
- **How slot 0 learns to fire for cats = training**: give cat photo → target [1,0,0], dog → [0,1,0], adjust weights via backprop → the rule "cat-like color pattern → slot 0 high" gets etched into the weights (the thing you did for XOR).
- **The bridge colors→cat = layered features**: color→edges→textures→parts (ears, eyes)→"these parts together = cat" (the ears/nose/fur in the big-picture diagram).
- Summary: the output does follow the input (colors) but **through the trained weights**. The name "cat" didn't come from the colors — **we pinned it to slot 0**. Retrain with swapped labels and a different slot fires, same architecture.

## 136. Same Code, but Random Init Makes Some Runs Good and Some Bad — True, but There Are Ways to Reduce the Luck

- **KIM's observation (correct)**: random weight init → same architecture & code still gives different results per run; good vs stuck models (generalizing note 125's "sometimes", note 129's B plateau).
- Refinement: we don't draw "good final weights" — we draw a **good starting point**. Training = rolling downhill. Good start → good valley; bad start → shallow ditch (plateau) and stops.
- **Ways to reduce the luck**:
  - **Smart initialization** (He/Xavier): not pure random but scaled random → signals don't vanish/explode → far fewer bad starts (real init isn't naive random).
  - **Run multiple seeds, keep the best on validation** (note 133).
  - **Better optimizers** (momentum/Adam) + **lr tuning**: escape plateaus/ditches.
- **Twist**: the smaller the problem (XOR + minimal neurons), the more luck matters (narrow valleys, many bad ditches). Large real nets have a more forgiving landscape where most starts converge to comparably good solutions (over-parameterization). The "coin flip" feel is strongest exactly in this XOR toy.

## 137. Why Big Nets Converge Well from Most Starting Points — High Dimensions Have Few Real Traps, and Good Solutions Are Everywhere

- **① Real local minima are rare in high dimensions**: a trap = every direction is uphill (can't leave). 1M weights = 1M dimensions → to be stuck, **all 1M directions must be uphill at once** (extremely unlikely). Usually at least one direction still slopes down → that point is a **saddle**, not a trap → gradient descent (+momentum) slips off and keeps descending.
- **② There isn't one good solution but countless ones**: with many weights, combos that fit the data are everywhere (note 130's redundancy = neurons are interchangeable). Not searching for one special needle — good valleys are all over → most starts descend into one of them. And those valleys mostly have similarly low loss → wherever you land, results are comparable.
- **Contrast**: XOR + minimal neurons = low-dimensional + the solution valley is narrow and rare → bad ditches are relatively big and common → luck matters a lot (note 136).
- Caveat: the full theory is still active research. The above are the well-accepted intuitions (saddles dominate in high-D; over-parameterization smooths optimization).

## 138. The Mechanism of a Line Failing to Reach the Answer — Samples' "Pulls" Fight and Cancel (Tug-of-War)

- KIM's intuition (core correct): several inputs pull the direction evenly so it doesn't lean one way → can't get there. ✓
- Refinement: rather than a hidden neuron "computing each input's probability and carrying it" — **each training sample pulls the line toward itself (that sample's gradient)**. One step's move = **the sum/average of all pulls**. (The final probability is at the output, notes 126·127.)
- **Key**: if samples pull in different directions, the sum cancels and shrinks → the line can't lean decisively and stalls (a taut tug-of-war). AND 3/4: the wrong [0,1] pulls, while the 3 correct points pull back "stay put" → sum ≈ 0.
- Link to note 129 A vs B: "not leaning (sum ≈ 0)" happens two ways — **A**: each pull is already 0 (all satisfied, loss ≈ 0, good stop). **B**: each pull is big but they fight and cancel (loss high, stuck). **Failing to reach the answer = B (fighting)**.
- Separate cause: saturated/dead neurons kill the pull itself to 0 (notes 125·129 B). KIM's explanation nails the "cancellation" case.

## 139. Input Normalization — Why (input multiplies the gradient) + How (standardize: mean0/std1)

- **Why** (KIM's derivation): gradient = input × 2 × (pred − answer) (note 129), so a **large-value feature gets a large pull (gradient)** → it dominates learning and moves fast. Conversely a small feature (0.2) is barely pulled → crawls. → **imbalance**: no single lr suits both (tune for the big one and the small one won't learn; tune for the small one and the big one diverges). Large values also raise z → saturation risk (note 129 B).
- **How**: **standardization (z-score)** — each feature: (value − that feature's mean) ÷ that feature's std → every feature mean0/std1. Equal scale → fair pulls → stable learning under one lr.
- **Caution (wired at ⑤)**: compute mean/std from **training data only** and apply the same to test. Test must be "unseen" (note 130), so peeking at test stats is cheating (data leakage).
- **What mean/std are**: mean = sum ÷ count. std (standard deviation) = spread around the mean = "typical distance from the mean" = sqrt(mean((x−mean)²)). `(x−mean)` → center 0, `÷std` → spread 1. `axis=0` = compute per column/feature (4 features → 4 results). Iris mean≈[5.84,3.06,3.76,1.2], std≈[.83,.43,1.76,.76]. Ex [2,4,6]→mean4·std1.63→standardized[-1.22,0,1.22].

---

## The Big Picture at a Glance

```
[Data prep]    images on disk + answers
      │  (CPU preprocessing: convert to numbers, labeling)
      ▼
[Load to VRAM] model weights (resident) + batch (images + answers, swapped)
      │
      ▼
[GPU compute]  forward pass (predict) → loss (compare with answer) → backprop (correct weights)
      │  (thousands of cores do matrix multiplication in parallel, using cache and registers)
      ▼
[Repeat]       swap batches, tens of thousands of times → weights complete = training
      │
      ▼
[Inference]    use the fixed weights to answer new inputs (real use)
```

**Inside one neural-network layer (the sets / intermediate-values view)**

```
measurements [2,3,1]
   │  layer 1: several sets (each a different feature) → each set makes 1 intermediate value
   ▼           (multiply and sum → bend with ReLU)
intermediate values [6, 3, 8, ...]   ← features (ears, nose, fur …)
   │  layer 2: take the previous intermediate values as input, multiply and sum again → ReLU
   ▼
intermediate values [...]  → … → last layer (fitted to # of outputs) → final score = answer
```

---

### Topics Worth Digging Into Further (not yet covered)
- Why training uses up to several times the model's size in memory (gradients, optimizer state)
- How convolution detects edges (filter visualization)
- self-attention (the heart of the transformer) — THE CRUX of the nanoGPT lecture
- Concrete techniques to prevent overfitting (more data, dropout, etc.)
- Taking yesterday's Iris code (decision tree) and re-running it as a neural network
- Building a small LLM from the ground up with nanoGPT (Karpathy's lecture + build-nanogpt)
- Training a thinking model yourself with reinforcement learning (RLVR) — the "learn to add two numbers" practice problem
- Virtual memory — a separate concept easily confused with VRAM

### The Next Practice Steps (time to get hands-on)
- ✅ Converting Day 2 (regression) to PyTorch — done (`regression_pytorch.py`) — saw the 4-line training loop firsthand
- In `regression_pytorch.py`, change the learning rate (lr) to 0.01/2.0 to see convergence/divergence firsthand
- In `verify_flower.py` too, change the learning rate to confirm divergence/convergence
- Verify that even when you change the inputs and answers, `gradient = measurement × 2 × (prediction − answer)` holds
- **Next**: Day 3 (model comparison, overfitting, cross-validation), or neural networks (add layers and insert ReLU)
- Increase the layers 1→2→3 and plot how the decision boundary grows more complex (see the activation-function effect with your own eyes)
- Karpathy's "Let's build GPT" video: chunk 1 (review) fast → chunk 2 (self-attention) slowly
