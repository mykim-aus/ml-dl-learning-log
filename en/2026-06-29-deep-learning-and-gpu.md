<sub>🌐 **English** · [한국어](../ko/2026-06-29-딥러닝과-GPU.md)</sub>

# Deep Learning & GPU Study Notes

> Machine learning basics → GPU hardware → how memory works → the actual training process → the principles behind weights and derivatives → LLMs and recent trends:
> these are notes that pull a single session's worth of conversation into one continuous thread.

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
