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
