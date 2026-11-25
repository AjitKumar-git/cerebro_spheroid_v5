# Notes

## Types of Reasoning in Artificial Intelligence
*Source: GeeksforGeeks (Last Updated: 27 May, 2026) | Added: 2026-11-25*

In AI systems, **reasoning** is the process of using available information and logical methods to make decisions, solve problems, and generate conclusions. Different types of reasoning enable AI systems to learn from data, handle uncertainty, and respond intelligently in dynamic real-world situations.

### 1. Deductive Reasoning
- **Definition**: Starts with general principles and applies them to specific cases to arrive at certain conclusions. If the premises are true, the conclusion must also be true. ([Reference](https://www.geeksforgeeks.org/artificial-intelligence/deductive-reasoning-in-ai/))
- **Example**: If all humans are mortal and Socrates is a human, then Socrates is mortal.
- **Application**: Used in expert systems where predefined rules help derive accurate decisions, such as identifying diseases from known symptom patterns.

### 2. Inductive Reasoning
- **Definition**: Works by drawing general conclusions from specific observations. The conclusions reached are not certain but are based on probability and patterns observed in the data. ([Reference](https://www.geeksforgeeks.org/artificial-intelligence/inductive-reasoning-in-ai/))
- **Example**: If we observe that the sun rises in the east every day, we might infer that it will rise in the east tomorrow.
- **Application**: Widely used in machine learning systems to identify patterns and make predictions, such as movie recommendation systems.

### 3. Abductive Reasoning
- **Definition**: Starts with incomplete observations and seeks the most likely explanation. It is about making educated guesses based on available data, even if not all facts are known. ([Reference](https://www.geeksforgeeks.org/artificial-intelligence/abductive-reasoning-in-ai/))
- **Example**: If a patient has a fever and cough, a doctor might hypothesize that they have the flu even though other illnesses could cause similar symptoms.
- **Application**: Used in diagnostic AI systems to predict probable causes of problems or diseases from available symptoms or data.

### 4. Analogical Reasoning
- **Definition**: Involves comparing two situations that are similar and using knowledge from one to solve problems in another. It helps AI systems solve problems in new domains by applying solutions from related areas.
- **Example**: If flying a helicopter is similar to flying a drone, knowledge from piloting a helicopter can be transferred to flying a drone.
- **Application**: Used in robotics and cognitive systems where AI transfers solutions from previously solved tasks to similar new tasks.

### 5. Common Sense Reasoning
- **Definition**: Allows AI systems to make decisions using everyday knowledge and practical understanding of the world. It involves making judgments about the world that are obvious to humans but difficult for machines to understand.
- **Example**: If it rains, we can expect the ground to get wet even without explicitly stating it.
- **Application**: Used in conversational AI and autonomous systems to interpret everyday situations and respond naturally.

### 6. Monotonic Reasoning
- **Definition**: A form of reasoning where conclusions, once drawn, cannot be reversed even if new information becomes available. This ensures that conclusions remain consistent regardless of updates to the knowledge base.
- **Example**: The statement *"The Sahara is a desert"* remains true even if more information about the world's deserts is introduced.
- **Application**: Used in formal verification and rule-based systems where consistent conclusions are required.

### 7. Nonmonotonic Reasoning
- **Definition**: In contrast to monotonic reasoning, allows AI systems to revise their conclusions based on new information. Important for decision-making in dynamic and unpredictable environments. ([Reference](https://www.geeksforgeeks.org/artificial-intelligence/monotonic-reasoning-vs-non-monotonic-reasoning/))
- **Example**: Initially, we might think that all birds can fly, but we revise this conclusion after learning about penguins, which cannot fly.
- **Application**: Used in adaptive AI systems like autonomous vehicles and traffic management systems that continuously update decisions using new data.

### 8. Fuzzy Reasoning
- **Definition**: Deals with uncertainty by allowing for degrees of truth rather than binary true/false values. Useful when data is vague or incomplete. ([Reference](https://www.geeksforgeeks.org/artificial-intelligence/fuzzy-logic-introduction/))
- **Example**: The statement *"It's warm outside"* is vague. Fuzzy reasoning might assign a value like `0.7` to represent how warm it is.
- **Application**: Used in smart appliances and autonomous systems where decisions must be made under uncertain or imprecise conditions.
