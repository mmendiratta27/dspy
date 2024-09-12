

import sys
import os

try: # When on google Colab, let's clone the notebook so we download the cache.
    import google.colab  # noqa: F401
    repo_path = 'dspy'
    os.system("git -C $repo_path pull origin || git clone https://github.com/stanfordnlp/dspy $repo_path")
except:
    repo_path = '.'

if repo_path not in sys.path:
    sys.path.append(repo_path)

# Set up the cache for this notebook
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(repo_path, 'cache')

import pkg_resources # Install the package if it's not installed
if "dspy-ai" not in {pkg.key for pkg in pkg_resources.working_set}:
    os.system("pip3 install -U pip")
    os.system("pip3 install dspy-ai")
    os.system("pip3 install openai~=0.28.1")
    # !pip install -e $repo_path

import dspy
os.system("ollama pull phi3")
ollama_phi3 = dspy.OllamaLocal(model='phi3')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='https://en.wikipedia.org/wiki/The_Avengers_(2012_film)')

dspy.settings.configure(lm=ollama_phi3, rm=colbertv2_wiki17_abstracts)

class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Define the predictor.
generate_answer = dspy.Predict(BasicQA)

question = "Who are the strongest Avengers?"


# Define the predictor. Notice we're just changing the class. The signature BasicQA is unchanged.
generate_answer_with_chain_of_thought = dspy.ChainOfThought(BasicQA)

# Call the predictor on the same input.
pred = generate_answer_with_chain_of_thought(question=question)


# Print the input and the prediction.
print(f"Question: {question}")
print(f"Thought: {pred.rationale}")
print(f"Chain of Thought Answer: {pred.answer}")

# Define the predictor.
generate_answer = dspy.Predict(BasicQA)

# Call the predictor on a particular input.
pred = generate_answer(question=question)

print(f"Regular Answer: {pred.answer}")


